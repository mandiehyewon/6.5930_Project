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

    # write map space constraints
    dataflow_dir = "_dataflows"
    pe_test_path = os.path.join(dataflow_dir, "pe_test.yaml")
    if not os.path.exists(dataflow_dir):
        os.makedirs(dataflow_dir)     
    with open(pe_test_path, "w+") as f:
        f.write(_get_pe_test_yaml())
    pe_test_path = PosixPath(pe_test_path)
    
    ### Parameters
    channel_sizes=[8,16,24,32,16]
    P = [3,3,1,1]
    Q = [3,3,1,1]
    F = [218,109,109,55]  
    pe_shapes = [(1,16),(2,8),(4,4)] 
    
    results = {}
    for shape in pe_shapes:
        for fs in filter_sizes:
            for i in range(len(channel_sizes)-1):
                layer_path = os.path.join(dataflow_dir, f"layer{i}_fs{fs}_{shape[0]}x{shape[1]}.yaml")
                with open(layer_path, "w+") as f:
                    f.write(_get_layer(fs, channel_sizes[i], channel_sizes[i+1], P[i] ,Q[i], F[i], shape))
                layer_path = PosixPath(layer_path)
                hw_arch_path = os.path.join(hw_base_path, f'system_arch_{shape[0]}x{shape[1]}.yaml')
                hw_arch_path = PosixPath(hw_arch_path)
                stats, mapping = run_timeloop_mapper(hw_arch_path, hw_components_dir_path, layer_path, pe_test_path, mapper_config_path)
                layer_energy = float(stats.split("\n")[-22].split(" ")[1])
                layer_macs = int(stats.split("\n")[-18].split(" ")[-1])
                layer_cycles = int(stats.split("\n")[-23].split(" ")[-1])
                results[f"layer{i}_fs{fs}_{shape[0]}x{shape[1]}.yaml"] = (layer_energy, layer_macs, layer_cycles)
                print(layer_energy, layer_macs, layer_cycles)
 
    print(results)
    return results

'''
def _get_pe_test_yaml():
    mapspace = f"""
mapspace:
  targets:
    - name: DRAM
      type: bypass
      bypass:
    - name: DRAM
      type: temporal
      factors:
      permutation:
    - name: global_buffer
      type: spatial
      factors:
      permutation:
    - name: global_buffer
      type: temporal
      factors:
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
      bypass: [Weights, Inputs]"""

    return mapspace
'''


def _get_pe_test_yaml():
    
    mapspace = f"""
mapspace:
  targets:
    - name: DRAM
      type: bypass
      keep: [Inputs, Outputs, Weights]
      bypass: []
    - name: global_buffer
      type: bypass
      keep: []
      bypass: []
    - name: global_buffer
      type: temporal
      factors: P=1 Q=1 R=1 S=1
      permutation: #SRQPCMN
    - name: global_buffer
      type: spatial
      factors: P=1 Q=1 R=1 S=1 N=1
      permutation: #CMRSTPQFN
      split: 1
    - name: scratchpad
      type: bypass
      keep: [Weights]
      bypass: [Inputs, Outputs]
    - name: scratchpad
      type: temporal
      factors: R=0 S=0 P=0 Q=0
      permutation: #QPSR
    - target: weights_reg
      type: bypass
      keep: [Weights]
      bypass: [Inputs, Outputs]
    - target: weights_reg
      type: temporal
      factors: R=1 S=1 T=1 P=1 Q=1 F=1 N=1 C=1 M=1
    - target: input_activation_reg
      type: bypass
      keep: [Inputs]
      bypass: [Weights, Outputs]
    - target: input_activation_reg
      type: temporal
      factors: R=1 S=1 T=1 P=1 Q=1 F=1 N=1 C=1 M=1
    - target: output_activation_reg
      type: bypass
      keep: [Outputs]
      bypass: [Weights, Inputs]
    - target: output_activation_reg
      type: temporal
      factors: R=1 S=1 T=1 P=1 Q=1 F=1 N=1 C=1 M=1"""
    
    return mapspace

def _get_layer(filter_size, channels_in, channels_out, P, Q, F, shape):

    pe_x = shape[0]
    pe_y = shape[1]
    
    # Do padding to best match spatial arch using heuristics
    
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
    N: 8 # batch size
    C: {c_in} # in channels (number of kernels)
    M: {c_out} # out channels (number of kernels)
    R: {fs} # filter width
    S: {fs} # filter height
    T: {fs} # filter depth
    Q: {Q} # output image width (x) try to make same number of macs
    P: {P} # output image height (y)
    F: {F} # output image depth (z)
    Wstride: 1
    Hstride: 1
    Dstride: 1"""

    return layer_spec

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