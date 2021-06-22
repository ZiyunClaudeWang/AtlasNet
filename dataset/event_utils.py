import torch
import numpy as np

def calc_floor_ceil_delta(x): 
    '''
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]
    '''

    x_fl = torch.floor(x)
    x_ce = x_fl + 1

    dx_ce = x - x_fl
    dx_fl = x - x_ce

    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]

def create_update(x, y, t, dt, p, vol_size, device="cpu"):
    '''
    assert (x>=0).byte().all() 
    assert (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all()
    assert (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() 
    '''
    #assert (t<vol_size[0] // 2).byte().all()

    ''''
    if not (t < vol_size[0] // 2).byte().all():
        print(t[t >= vol_size[0] // 2])
        print(vol_size)
        raise AssertionError()
    '''

    vol_mul = torch.where(p < 0,
                          torch.ones(p.shape, dtype=torch.long).to(device) * vol_size[0] // 2,
                          torch.zeros(p.shape, dtype=torch.long).to(device))

    inds = (vol_size[1]*vol_size[2]) * (t + vol_mul)\
         + (vol_size[2])*y\
         + x

    vals = dt

    return inds, vals

def gen_discretized_event_volume(x, y, t, p, vol_size, device="cpu"):
    # volume is [t, x, y]
    # events are Nx4
    npts = x.shape[0]
    volume = torch.zeros(vol_size)

    t_min = t.min()
    t_max = t.max()
    t_scaled = torch.clip((t-t_min) / (t_max - t_min) - 1e-5, 0) * (vol_size[0] // 2-1)

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     p,
                                     vol_size,
                                     device=device)
    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],
                                     p,
                                     vol_size,
                                     device=device)

    volume.view(-1).put_(torch.stack((inds_fl, inds_ce)), 
                        torch.stack((vals_fl, vals_ce)), accumulate=True)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)
    return volume
 
def normalize_event_volume(event_volume):
    event_volume_flat = event_volume.view(-1)
    nonzero = torch.nonzero(event_volume_flat)
    nonzero_values = event_volume_flat[nonzero]
    if nonzero_values.shape[0]:
        lower = torch.kthvalue(nonzero_values,
                               max(int(0.02 * nonzero_values.shape[0]), 1),
                               dim=0)[0][0]
        upper = torch.kthvalue(nonzero_values,
                               max(int(0.98 * nonzero_values.shape[0]), 1),
                               dim=0)[0][0]
        max_val = max(abs(lower), upper)
        event_volume = torch.clamp(event_volume, -max_val, max_val)
        event_volume /= max_val
    return event_volume
