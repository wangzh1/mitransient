from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitransient.integrators.common import TransientRBIntegrator, mis_weight

from mitsuba import Log, LogLevel
from mitsuba.math import ShadowEpsilon  # type: ignore
from typing import Tuple, Optional


class TransientNLOSPath(TransientRBIntegrator):
    # TODO(diego): docs
    r"""
    .. _integrator-prb:

    Path Replay Backpropagation (:monosp:`prb`)
    -------------------------------------------

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: 6)

     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (Default: 5)

    This plugin implements a basic Path Replay Backpropagation (PRB) integrator
    with the following properties:

    - Emitter sampling (a.k.a. next event estimation).

    - Russian Roulette stopping criterion.

    - No reparameterization. This means that the integrator cannot be used for
      shape optimization (it will return incorrect/biased gradients for
      geometric parameters like vertex positions.)

    - Detached sampling. This means that the properties of ideal specular
      objects (e.g., the IOR of a glass vase) cannot be optimized.

    See ``prb_basic.py`` for an even more reduced implementation that removes
    the first two features.

    See the papers :cite:`Vicini2021` and :cite:`Zeltner2021MonteCarlo`
    for details on PRB, attached/detached sampling, and reparameterizations.

    .. tabs::

        .. code-tab:: python

            'type': 'prb',
            'max_depth': 8
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        self.filter_depth: int = props.get('filter_depth', -1)
        if self.filter_depth != -1 and self.max_depth != -1:
            if self.filter_depth >= self.max_depth:
                Log(LogLevel.Warn,
                    'You have set filter_depth >= max_depth. '
                    'This will cause the final image to be all zero.')
        self.discard_direct_paths: bool = props.get(
            'discard_direct_paths', False)
        self.laser_sampling: bool = props.get(
            'nlos_laser_sampling', False)
        self.hg_sampling: bool = props.get(
            'nlos_hidden_geometry_sampling', False)
        self.hg_sampling_do_rroulette = (
            props.get('nlos_hidden_geometry_sampling_do_rroulette', False)
            and
            self.hg_sampling
        )
        # NOTE: turned off for now, pdf does weird things and have not figured
        # it out yet
        self.hg_sampling_do_rs: bool = False
        self.hg_sampling_includes_relay_wall: bool = (
            props.get('nlos_hidden_geometry_sampling_includes_relay_wall', False)
            and
            self.hg_sampling
        )

    def prepare_transient(self, scene: mi.Scene, sensor: mi.Sensor):
        super().prepare_transient(scene, sensor)

        # prepare laser sampling

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        if self.laser_sampling:
            trafo: mi.Transform4f = sensor.emitter.world_transform()
            laser_origin: mi.Point3f = trafo.translation()
            laser_dir: mi.Vector3f = trafo.transform_affine(
                mi.Vector3f(0, 0, 1))
            si = scene.ray_intersect(mi.Ray3f(laser_origin, laser_dir))
            assert dr.all(
                si.is_valid()), 'The emitter is not pointing at the scene!'
            self.nlos_laser_target: mi.Point3f = si.p

        # prepare hidden geometry sampling

        self.hidden_geometries = scene.shapes_dr()
        self.hidden_geometries_distribution = []
        for shape in scene.shapes_dr():
            is_relay_wall = shape.sensor() == sensor
            if not self.hg_sampling_includes_relay_wall and is_relay_wall:
                self.hidden_geometries_distribution.append(0)
                continue
            self.hidden_geometries_distribution.append(shape.surface_area()[0])

        self.hidden_geometries_distribution = mi.DiscreteDistribution(
            self.hidden_geometries_distribution)

    def _sample_hidden_geometry_position(
            self, ref: mi.Interaction3f,
            sample2: mi.Point2f, active: mi.Mask) -> mi.PositionSample3f:
        """
        For non-line of sight scenes, sample a point in the hidden
        geometry's surface area. The "hidden geometry" is defined
        as any object that does not contain an \ref nloscapturemeter
        plugin (i.e. every object but the relay wall)

        Parameters
        ----------
        ref
            A reference point somewhere within the scene

        sample2
            A uniformly distributed 2D vector

        active
            A boolean mask

        Returns
        -------
        Position sampling record
        """

        if len(self.hidden_geometries) == 0:
            return dr.zeros(mi.PositionSample3f)

        if len(self.hidden_geometries) == 1:
            return self.hidden_geometries[0].sample_position(
                ref.time, sample2, active)

        index, new_sample, shape_pdf = \
            self.hidden_geometries_distribution.sample_reuse_pmf(
                sample2.x, active)
        sample2.x = new_sample

        shape: mi.ShapePtr = dr.gather(
            mi.ShapePtr, self.hidden_geometries, index, active)
        ps = shape.sample_position(ref.time, sample2, active)
        ps.pdf *= shape_pdf

        return ps

    def emitter_nee_sample(
            self, mode: dr.ADMode, scene: mi.Scene, sampler: mi.Sampler,
            si: mi.SurfaceInteraction3f, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext,
            β: mi.Spectrum, distance: mi.Float, η: mi.Float, depth: mi.UInt,
            extra_weight: mi.Float, active_e: mi.Mask, add_transient) -> mi.Spectrum:
        ds, emitter_spec = scene.sample_emitter_direction(
            ref=si, sample=sampler.next_2d(active_e), test_visibility=True, active=active_e)
        active_e &= dr.neq(ds.pdf, 0.0)

        primal = mode == dr.ADMode.Primal
        with dr.resume_grad(when=not primal):
            if not primal:
                # Given the detached emitter sample, *recompute* its
                # contribution with AD to enable light source optimization
                ds.d = dr.normalize(ds.p - si.p)
                emitter_val = scene.eval_emitter_direction(si, ds, active_e)
                emitter_spec = dr.select(
                    dr.neq(ds.pdf, 0), emitter_val / ds.pdf, 0)
                dr.disable_grad(ds.d)

            # Query the BSDF for that emitter-sampled direction
            wo = si.to_local(ds.d)
            bsdf_spec, bsdf_pdf = bsdf.eval_pdf(
                ctx=bsdf_ctx, si=si, wo=wo, active=active_e)

            mis = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

            Lr_dir = mi.Spectrum(0)
            active_e &= (dr.eq(depth, self.filter_depth)
                         |
                         ~(self.discard_direct_paths & (depth < 2)))
            Lr_dir[active_e] = β * mis * bsdf_spec * emitter_spec

            effective_weight = dr.select(dr.any(Lr_dir > 0), extra_weight, 0)
            add_transient(Lr_dir, effective_weight, distance + ds.dist * η,
                          si.wavelengths, active_e)

        return Lr_dir

    def emitter_laser_sample(
            self, mode: dr.ADMode, scene: mi.Scene, sampler: mi.Sampler,
            si: mi.SurfaceInteraction3f, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext,
            β: mi.Spectrum, distance: mi.Float, η: mi.Float, depth: mi.UInt,
            extra_weight: mi.Float, active_e: mi.Mask, add_transient) -> mi.Spectrum:
        """
        NLOS scenes only have one laser emitter - standard
        emitter sampling techniques do not apply as most
        directions do not emit any radiance, it needs to be very
        lucky to bsdf sample the exact point that the laser is
        illuminating

        this modifies the emitter sampling so instead of directly
        sampling the laser we sample(1) the point that the laser
        is illuminating and then(2) the laser
        """
        # TODO probably needs some "with dr.resume_grad(when=not primal):"
        primal = mode == dr.ADMode.Primal

        # 1. Obtain direction to NLOS illuminated point
        #    and test visibility with ray_test
        d = self.nlos_laser_target - si.p
        distance_laser = dr.norm(d)
        d /= distance_laser
        ray_bsdf = mi.Ray3f(o=si.p, d=d,
                            maxt=distance_laser * (1.0 - ShadowEpsilon),
                            time=si.time,
                            wavelengths=si.wavelengths)
        active_e &= ~scene.ray_test(ray_bsdf, active_e)

        # 2. Evaluate BSDF to desired direction
        wo = si.to_local(d)
        bsdf_spec = bsdf.eval(ctx=bsdf_ctx, si=si, wo=wo, active=active_e)
        bsdf_spec = si.to_world_mueller(bsdf_spec, -wo, si.wi)

        ray_bsdf.maxt = dr.inf
        si_bsdf: mi.SurfaceInteraction3f = scene.ray_intersect(
            ray_bsdf, active_e)
        active_e &= si_bsdf.is_valid()
        active_e &= dr.any(mi.depolarizer(bsdf_spec) > dr.epsilon(mi.Float))
        wl = si_bsdf.to_local(-d)
        active_e &= mi.Frame3f.cos_theta(wl) > 0.0

        # NOTE(diego): as points are not randomly chosen,
        # we need to account for d^2 and cos term because of
        # the solid angle projection of si.p to
        # nlos_laser_target.
        # This is like a point light, but the extra cos term
        # exists as it is not a point light that emits in all
        # directions :^)
        # The incident cos term at
        # nlos_laser_target will be taken into account by
        # emitter_nee_sample's bsdf
        bsdf_spec *= dr.sqr(dr.rcp(distance_laser)) * mi.Frame3f.cos_theta(wl)

        bsdf_next = si_bsdf.bsdf(ray=ray_bsdf)

        # 3. Combine laser + emitter sampling
        return self.emitter_nee_sample(
            mode=mode, scene=scene, sampler=sampler, si=si_bsdf,
            bsdf=bsdf_next, bsdf_ctx=bsdf_ctx, β=β * bsdf_spec, distance=distance + distance_laser * η, η=η,
            depth=depth+1, extra_weight=extra_weight, active_e=active_e, add_transient=add_transient)

    def hidden_geometry_sample(
            self, scene: mi.Scene, sampler: mi.Sampler, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
            _: mi.Float, sample2: mi.Point2f, active: mi.Mask) -> Tuple[mi.BSDFSample3f, mi.Spectrum]:

        # first attempt at sampling a visible position
        active_original = mi.Mask(active)
        ps_hg: mi.PositionSample3f = self._sample_hidden_geometry_position(
            ref=si, sample2=sample2, active=active)
        active &= dr.neq(ps_hg.pdf, 0.0)
        # check visibility
        d = ps_hg.p - si.p
        dist = dr.norm(d)
        d /= dist
        ray_hg = si.spawn_ray(d)
        si_hg: mi.SurfaceInteraction3f = scene.ray_intersect(ray_hg, active)
        active &= si_hg.is_valid() & (dr.dot(si.n, d) > 0.0)

        # try again for the lanes that do not have a visible position
        extra_weight_i = dr.zeros(mi.UInt32)
        if self.hg_sampling_do_rs:
            iters_rs = dr.zeros(mi.UInt32)
            max_rs_iterations = 10
            retry_rs = active ^ active_original
            loop_rs = mi.Loop(name='Hidden geometry rejection sampling',
                        state=lambda: (ps_hg, si_hg, ray_hg, d, sampler, sample2, retry_rs, active, iters_rs))
            loop_rs.set_max_iterations(max_rs_iterations)
            while loop_rs(retry_rs):
                iters_rs[retry_rs] += 1
                sample2 = mi.Point2f(dr.select(retry_rs, sampler.next_2d(retry_rs), sample2))
                ps_hg = dr.select(
                    retry_rs,
                    self._sample_hidden_geometry_position(
                        ref=si, sample2=sample2, active=retry_rs),
                    ps_hg)
                successful_retries = retry_rs & dr.neq(ps_hg.pdf, 0.0)
                # check visibility
                d = ps_hg.p - si.p
                dist = dr.norm(d)
                d /= dist
                ray_hg = si.spawn_ray(d)
                si_hg: mi.SurfaceInteraction3f = scene.ray_intersect(
                    ray_hg, retry_rs)
                successful_retries &= si_hg.is_valid() & \
                    (dr.dot(si.n, d) > 0.0)
                active |= successful_retries
                retry_rs &= (~successful_retries) & (iters_rs < max_rs_iterations)
            extra_weight_i = dr.select(iters_rs < max_rs_iterations, iters_rs, 0.0)

        # compute the PDF of the position sample (num_intersections)
        si_test = mi.SurfaceInteraction3f(dr.detach(si_hg))
        active_test = mi.Mask(active)
        num_intersections = mi.UInt(0)
        loop = mi.Loop(name='Hidden geometry pdf calculation',
                       state=lambda: (num_intersections, active_test, si_test))
        loop.set_max_iterations(100)
        while loop(active_test):
            num_intersections[active_test] += 1
            ray_test = si_test.spawn_ray(d)
            si_test: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray_test, active_test)
            active_test &= \
                (si_test.t > 2 * dr.epsilon(mi.Float)) & si_test.is_valid()
        active[num_intersections >= 100] = False

        wo = si.to_local(d)
        bsdf_spec = bsdf.eval(ctx=bsdf_ctx, si=si, wo=wo, active=active)
        bsdf_spec = si.to_world_mueller(bsdf_spec, -wo, si.wi)

        wg = si_hg.to_local(-d)
        travel_dist = dr.norm(si_hg.p - si.p)
        bsdf_spec *= dr.sqr(dr.rcp(travel_dist)) * mi.Frame3f.cos_theta(wg)

        # discard low travel dist, produces high variance
        # the intergator will use bsdf sampling instead
        active &= travel_dist > 0.5
        active &= dr.any(mi.depolarizer(bsdf_spec) > dr.epsilon(mi.Float))

        bs: mi.BSDFSample3f = dr.zeros(mi.BSDFSample3f)
        bs.wo = wo
        bs.pdf = ps_hg.pdf * num_intersections
        bs.eta = 1.0
        bs.sampled_type = mi.BSDFFlags.Reflection
        bs.sampled_component = 0

        bsdf_spec /= bs.pdf

        return bs, dr.select(active & (bs.pdf > dr.epsilon(mi.Float)), bsdf_spec, 0.0), extra_weight_i

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               max_distance: mi.Float,
               add_transient,
               **kwargs  # Absorbs unused arguments
               ) -> Tuple[mi.Spectrum,
                          mi.Bool,
                          mi.Spectrum]:
        """
        See ``TransientADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        # FIXME maybe rename to just "weight" when i figure it out
        extra_weight = mi.Float(1)                    # Path weight
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        # Differential/adjoint radiance
        δL = mi.Spectrum(δL if δL is not None else 0)
        β = mi.Spectrum(1)                            # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes
        # NOTE(diego): nloscapturemeter sets ray.time in optical path length (distance)
        distance = mi.Float(ray.time)                 # Distance of the path

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        if self.camera_unwarp:
            raise AssertionError('Use account_first_and_last_bounces instead')

        # Record the following loop in its entirety
        loop = mi.Loop(name="Path Replay Backpropagation (%s)" % mode.name,
                       state=lambda: (sampler, ray, depth, L, δL, β, η, active,
                                      prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                                      distance, extra_weight))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        if self.laser_sampling:
            emitter_sample_f = self.emitter_laser_sample
        else:
            emitter_sample_f = self.emitter_nee_sample

        while loop(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.

            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All,
                                         coherent=dr.eq(depth, 0))

            # Update distance
            distance += dr.select(active, si.t, 0.0) * η
            active &= distance < max_distance

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            with dr.resume_grad(when=not primal):
                Le = β * mis * ds.emitter.eval(si)

            # Add transient contribution
            effective_weight = dr.select(dr.any(Le > 0), extra_weight, 0)
            add_transient(Le, effective_weight, distance, ray.wavelengths, active)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_e = active_next & mi.has_flag(
                bsdf.flags(), mi.BSDFFlags.Smooth)

            # Uses NEE or laser sampling depending on self.laser_sampling
            Lr_dir = emitter_sample_f(
                mode, scene, sampler,
                si, bsdf, bsdf_ctx,
                β, distance, η, depth,
                extra_weight, active_e, add_transient)

            # ------------------ Detached BSDF sampling -------------------

            if self.hg_sampling and self.hg_sampling_do_rroulette:
                # choose HG or BSDF sampling with Russian Roulette
                hg_prob = mi.Float(0.5)
                do_hg_sample = sampler.next_1d(active) < hg_prob
                pdf_bsdf_method = dr.select(
                    do_hg_sample,
                    hg_prob,
                    mi.Float(1.0) - hg_prob)
            else:
                # only one option
                do_hg_sample = mi.Mask(self.hg_sampling)
                pdf_bsdf_method = mi.Float(1.0)

            active_hg = active_next & do_hg_sample
            bsdf_sample_hg, bsdf_weight_hg, extra_weight_i = self.hidden_geometry_sample(
                scene, sampler, bsdf,
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_hg)

            hg_failed = dr.all(mi.depolarizer(bsdf_weight_hg)
                               < dr.epsilon(mi.Float))
            add_extra_weights = active_hg & (~hg_failed)
            extra_weight = dr.select(
                add_extra_weights,
                # NOTE transient_hdr_film subtracts one from extra_weight
                extra_weight * (extra_weight_i + 1),
                extra_weight)

            active_nhg = active_next & (~do_hg_sample | hg_failed)
            extra_weight[active_hg & hg_failed] += 1.0

            bsdf_sample_nhg, bsdf_weight_nhg = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_nhg)
            bsdf_sample = dr.select(
                active_nhg, bsdf_sample_nhg, bsdf_sample_hg)
            bsdf_weight = dr.select(
                active_nhg, bsdf_weight_nhg, bsdf_weight_hg)

            # ---- Update loop variables based on current interaction -----

            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight / pdf_bsdf_method

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)
            active_next &= rr_prob > 0

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob) & (rr_prob > 0)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # ------------------ Differential phase only ------------------

            if not primal:
                with dr.resume_grad():
                    # 'L' stores the indirectly reflected radiance at the
                    # current vertex but does not track parameter derivatives.
                    # The following addresses this by canceling the detached
                    # BSDF value and replacing it with an equivalent term that
                    # has derivative tracking enabled. (nit picking: the
                    # direct/indirect terminology isn't 100% accurate here,
                    # since there may be a direct component that is weighted
                    # via multiple importance sampling)

                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si.to_local(ray.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(dr.neq(bsdf_val_det, 0),
                                                 dr.rcp(bsdf_val_det), 0)

                    # Differentiable version of the reflected indirect
                    # radiance. Minor optional tweak: indicate that the primal
                    # value of the second term is always 1.
                    Lr_ind = L * \
                        dr.replace_grad(1, inv_bsdf_val_det * bsdf_val)

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_dir + Lr_ind

                    if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            depth[si.is_valid()] += 1
            active = active_next

        return (
            L if primal else δL,  # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L                    # State for the differential phase
        )


mi.register_integrator("transient_nlos_path",
                       lambda props: TransientNLOSPath(props))
