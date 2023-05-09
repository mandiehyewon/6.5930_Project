import os
from loaders import run_timeloop_mapper
from pathlib import PosixPath
import math

def get_energy(filter_sizes=[3,5,7,9]):
    
    hw_base_path="designs/system_manual/arch/"
    hw_components_dir_path="designs/system_manual/arch/components"
    mapper_config_path = "designs/system_auto/mapper/mapper.yaml"

    # paths to Posix paths (required by loaders functions)
    hw_components_dir_path = PosixPath(hw_components_dir_path)
    mapper_config_path = PosixPath(mapper_config_path)

    dataflow_dir = "_dataflows"
    constraint_dir = "_constraints"
    
    ### Parameters
    channel_sizes=[1,8,16,24,32]
    P = [3,3,1,1]
    Q = [3,3,1,1]
    F = [218,109,109,55]  
    pe_shapes = [(1,16),(2,8),(4,4)] 
    
    results = {}
    for p in range(len(pe_shapes)):
        shape = pe_shapes[p]
        for fs in filter_sizes:
            for i in range(len(channel_sizes)-1):
                print('------------------------------------------')
                print(f'shape {shape}, filter {fs}, layer {i}')

                constraint_path = os.path.join(constraint_dir, f'mapping_layer{i}_fs{fs}_{shape[0]}x{shape[1]}.yaml')
                with open(constraint_path, "w+") as f:
                    f.write(_get_constraint(shape,fs,channel_sizes[i+1]))
                constraint_path = PosixPath(constraint_path)

                layer_path = os.path.join(dataflow_dir, f"layer{i}_fs{fs}_{shape[0]}x{shape[1]}.yaml")
                with open(layer_path, "w+") as f:
                    f.write(_get_layer(fs, channel_sizes[i], channel_sizes[i+1], P[i] ,Q[i], F[i], shape))
                layer_path = PosixPath(layer_path)
                hw_arch_path = os.path.join(hw_base_path, f'system_arch_{shape[0]}x{shape[1]}.yaml')
                hw_arch_path = PosixPath(hw_arch_path)
                stats, mapping = run_timeloop_mapper(hw_arch_path, hw_components_dir_path, layer_path, constraint_path, mapper_config_path)
                #print(stats)
                print(mapping)
                layer_energy = float(stats.split("\n")[-22].split(" ")[1])
                layer_macs = int(stats.split("\n")[-18].split(" ")[-1])
                layer_cycles = int(stats.split("\n")[-23].split(" ")[-1])
                pj_per_compute = float(stats.split("\n")[-3].split(" ")[-1])
                scratchpad = float(stats.split("\n")[-12].split(" ")[-1])
                global_buffer = float(stats.split("\n")[-11].split(" ")[-1])
                DRAM = float(stats.split("\n")[-10].split(" ")[-1])
                results[f"layer{i}_fs{fs}_{shape[0]}x{shape[1]}"] = (layer_energy, layer_macs, layer_cycles, pj_per_compute, scratchpad, global_buffer, DRAM)
                print(layer_energy, layer_macs, layer_cycles, pj_per_compute, scratchpad, global_buffer, DRAM)
                print('------------------------------------------')

    print(results)
    return results


def _get_constraint(shape, fs, channel_out):
    if shape == (1,16):
        mapspace = f"""
mapspace:
  targets:
    - name: DRAM
      type: bypass
      keep: [Inputs, Outputs, Weights]
      bypass: []
    - name: DRAM
      type: temporal
      factors: F=0 P=0 Q=0 C=0 T=0 #M=0 
      permutation: # MFPQ
    - name: global_buffer
      type: spatial
      factors: N=16
      permutation: #CMRSPQN
    - name: global_buffer
      type: temporal
      factors: R=0 S=0
      permutation: #CMRSPQN
    - name: scratchpad
      type: temporal
      factors: M=0
      permutation:
    - target: weights_reg
      type: bypass
      keep: [Weights]
      bypass: [Inputs, Outputs]
    - target: input_activation_reg
      type: bypass
      keep: [Inputs]
      bypass: [Weights, Outputs]
      permutation:
    - target: output_activation_reg
      type: bypass
      keep: [Outputs]
      bypass: [Weights, Inputs]
    """
    elif shape == (2,8):
        mapspace = f"""
mapspace:
  targets:
    - name: DRAM
      type: bypass
      keep: [Inputs, Outputs, Weights]
      bypass: []
    - name: DRAM
      type: temporal
      factors: F=0 P=0 Q=0 C=0 T=0 #M={channel_out/2} #M=20 #C=0 M=10 F=0
      permutation: # MFPQ
    - name: global_buffer
      type: spatial
      factors: N=8 M=2 #N=8
      permutation: #CMRSPQN
    - name: global_buffer
      type: temporal
      factors: R=0 S=0
      permutation: #CMRSPQN
    - name: scratchpad
      type: temporal
      factors: M={channel_out/2}
      permutation:
    - target: weights_reg
      type: bypass
      keep: [Weights]
      bypass: [Inputs, Outputs]
    - target: input_activation_reg
      type: bypass
      keep: [Inputs]
      bypass: [Weights, Outputs]
      permutation:
    - target: output_activation_reg
      type: bypass
      keep: [Outputs]
      bypass: [Weights, Inputs]
      """
    else:
        mapspace = f"""
mapspace:
  targets:
    - name: DRAM
      type: bypass
      keep: [Inputs, Outputs, Weights]
      bypass: []
    - name: DRAM
      type: temporal
      factors: F=0 P=0 Q=0 C=0 T=0 #M=0
      permutation: # MFPQ
    - name: global_buffer
      type: spatial
      factors: M=4 N=4 #T=4
      permutation: #CMRSPQN
    - name: global_buffer
      type: temporal
      factors: R=0 S=0
      permutation: #CMRSPQN
    - target: weights_reg
      type: bypass
      keep: [Weights]
      bypass: [Inputs, Outputs]
    - target: input_activation_reg
      type: bypass
      keep: [Inputs]
      bypass: [Weights, Outputs]
      permutation:
    - target: output_activation_reg
      type: bypass
      keep: [Outputs]
      bypass: [Weights, Inputs]
        """
    return mapspace
    

def _get_layer(filter_size, channels_in, channels_out, P, Q, F, shape, batch=16):

    # Do padding to best match spatial arch using heuristics
    
    pe_x = shape[0]
    pe_y = shape[1]
    
    if pe_x == 1 and pe_y == 16:
        R = filter_size
        S = filter_size
        T = filter_size
        c_in = channels_in
        c_out = channels_out
        P = P
        Q = Q
        F = F
        batch = _get_single_padded_size(16, batch)
    
    elif pe_x == 2 and pe_y == 8:
        R = filter_size
        S = filter_size
        T = filter_size
        c_in = channels_in
        c_out = _get_single_padded_size(2, channels_out)
        P = P
        Q = Q
        F = F
        batch = _get_single_padded_size(8, batch)
        
    elif pe_x == 4 and pe_y == 4:
        R = filter_size
        S = filter_size
        #T = _get_single_padded_size(4, filter_size)
        T = filter_size
        c_in = channels_in
        c_out = _get_single_padded_size(4, channels_out)
        P = P
        Q = Q
        F = F
        batch = _get_single_padded_size(4, batch)
                                    
    else:
        fs = _get_padded_size(shape, filter_size)
        c_in = _get_padded_size(shape, channels_in)
        c_out = _get_padded_size(shape, channels_out)
        P = _get_padded_size(shape, P)
        Q = _get_padded_size(shape, Q)
        F = _get_padded_size(shape, F)
            
    
    layer_spec = f"""
problem:
  shape:
    name: "Conv3D"
    dimensions: [ R, S, T, P, Q, F, C, M, N ]
    coefficients:
    - name: Wstride
      default: 1
    - name: Hstride
      default: 1
    - name: Dstride
      default: 1
    - name: Wdilation
      default: 1
    - name: Hdilation
      default: 1
    - name: Ddilation
      default: 1

    data-spaces:
    - name: Weights
      projection:
      - [ [C] ]
      - [ [M] ]
      - [ [R] ]
      - [ [S] ]
      - [ [T] ]
    - name: Inputs
      projection:
      - [ [N] ]
      - [ [C] ]
      - [ [R, Wdilation], [P, Wstride] ] # SOP form: RWdilation + PWstride
      - [ [S, Hdilation], [Q, Hstride] ] # SOP form: SHdilation + QHstride
      - [ [T, Ddilation], [F, Dstride] ] # SOP form: TDdilation + FDstride
    - name: Outputs
      projection:
      - [ [N] ]
      - [ [M] ]
      - [ [Q] ]
      - [ [P] ]
      - [ [F] ]
      read-write: True
# heuristic for padding 
  instance:
    N: {batch} # batch size
    C: {c_in} # in channels (number of kernels)
    M: {c_out} # out channels (number of kernels)
    R: {R} # filter width
    S: {S} # filter height
    T: {T} # filter depth
    Q: {Q} # output image width (x) try to make same number of macs
    P: {P} # output image height (y)
    F: {F} # output image depth (z)
    Wstride: 1
    Hstride: 1
    Dstride: 1"""

    return layer_spec

def _get_single_padded_size(factor, orig_size):
    if orig_size%factor==0:
        return orig_size
    else:
        return (math.floor(orig_size/factor)+1)*factor

def _get_padded_size(shape, orig_size):
    pe_x = shape[0]
    pe_y = shape[1]
    
    # Do padding to best match spatial arch using heuristics
    
    if pe_x == 1:
        if orig_size%pe_y==0:
            padded=orig_size
        else:
            padded = (math.floor(orig_size/pe_y)+1)*pe_y
        if padded-orig_size > orig_size/2:
            padded = orig_size
    else:
        if orig_size%pe_x==0 or orig_size%pe_y==0:
            padded = orig_size
        else:
            padded_x = (math.floor(orig_size/pe_x)+1)*pe_x
            padded_y = (math.floor(orig_size/pe_y)+1)*pe_y
            if padded_x > padded_y:
                padded = padded_y
            else:
                padded = padded_x
            if padded-orig_size > orig_size/2:
                padded = orig_size
            
    return padded