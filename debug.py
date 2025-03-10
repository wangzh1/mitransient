import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import mitransient as mitr
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import os


scene = mi.load_file(os.path.abspath('cornell-box/cbox_diffuse.xml'))
data_steady_ref, data_transient_ref = mi.render(scene, spp=1024)

params = mi.traverse(scene)
initial_vertex_pos = dr.unravel(mi.Point3f, params['light.vertex_positions'])

def apply_transformation(params, opt):
    
    trafo = mi.Transform4f().translate([opt['trans'].x, opt['trans'].y, 0.0])
    opt['trans'].y = dr.clip(opt['trans'].y, 0, 1)
    params['light.vertex_positions'] = dr.ravel(trafo @ initial_vertex_pos)
    params.update()

opt = mi.ad.Adam(lr=1)
opt['trans'] = mi.Point2f(50.0, 0.0)

apply_transformation(params, opt)
_, trans = mi.render(scene, params=params, spp=16)

loss_hist = []
for it in range(10000):
    apply_transformation(params, opt)
    image, trans = mi.render(scene, params=params, spp=16)

    loss = dr.mean(dr.square(trans - data_transient_ref)) * 100

    dr.backward(loss)

    opt.step()
    # import pdb; pdb.set_trace()
    loss_hist.append(loss)
    
    #save image
    if it % 100 == 0:
        mi.util.convert_to_bitmap(image).write(f"debug_output/transient_{it:03d}.png")
    print(f"Iteration {it:02d}: error={loss}, {opt['trans'].x}, {opt['trans'].y}", end='\r')
    