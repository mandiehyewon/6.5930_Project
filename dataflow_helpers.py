import os
from loaders import run_timeloop_mapper
from pathlib import PosixPath
import math

def get_energy(filter_sizes,
                num_channels=200,
                output_stationary=True,
                weight_stationary=True,
                hw_arch_path="designs/system_manual/arch/system_arch_1x16.yaml",
                hw_components_dir_path="designs/system_manual/arch/components",
                mapper_config_path = "designs/system_auto/mapper/mapper.yaml"):

    # Filter sizes should be divisible by 3
    # for f in filter_sizes:
    #     assert f % 3 == 0

    # paths to Posix paths (required by loaders functions)
    hw_arch_path = PosixPath(hw_arch_path)
    hw_components_dir_path = PosixPath(hw_components_dir_path)
    mapper_config_path = PosixPath(mapper_config_path)

    # more paths
    dataflow_dir = "_dataflows"
    out_stat_path = os.path.join(dataflow_dir, "output_stationary.yaml")
    w_stat_path = os.path.join(dataflow_dir, "weight_stationary.yaml")

    # get max filter size. we will assume a patch of max_fs x max_fs
    max_filter_size = max(filter_sizes)

    # write map space constraints
    if not os.path.exists(dataflow_dir):
        os.makedirs(dataflow_dir)

    # init result array & write output stationary mapspace constraints
    if output_stationary:
        out_energies = []
        with open(out_stat_path, "w+") as f:
            f.write(_get_output_stationary_yaml())
        out_stat_path = PosixPath(out_stat_path)

    # init result array & write weight stationary mapspace constraints
    if weight_stationary:
        w_energies = []
        with open(w_stat_path, "w+") as f:
            f.write(_get_weight_stationary_yaml())
        w_stat_path = PosixPath(w_stat_path)

    # get energies
    for fs in filter_sizes:
        layer_path = os.path.join(dataflow_dir, f"filtersize-{fs}.yaml")
        with open(layer_path, "w+") as f:
            f.write(_get_layer(fs, num_channels, max_filter_size))
        layer_path = PosixPath(layer_path)

        if output_stationary:
            stats, _ = run_timeloop_mapper(hw_arch_path, hw_components_dir_path, layer_path, out_stat_path, mapper_config_path)
            # print(stats)
            # print(stats.split("\n")[-3].split("= ")[-1])
            energy = float(stats.split("\n")[-3].split("= ")[-1])
            out_energies.append(energy)
        if weight_stationary:
            stats, _ = run_timeloop_mapper(hw_arch_path, hw_components_dir_path, layer_path, w_stat_path, mapper_config_path)
            energy = float(stats.split("\n")[-3].split("= ")[-1])
            w_energies.append(energy)

    # print
    print("filter size:\toutput stat:\tweight stat:")
    for i, fs in enumerate(filter_sizes):
        print(fs, end="\t\t")

        if output_stationary:
            print(f"{out_energies[i]}pJ/MACC", end="\t")
        else:
            print("\t\t")

        if weight_stationary:
            print(f"{w_energies[i]}pJ/MACC", end="")

            print()


def _get_output_stationary_yaml():

    mapspace = f"""
mapspace:
  targets:
    - name: DRAM
      type: bypass
      keep: [Inputs, Outputs, Weights]
      bypass: []
    - name: DRAM
      type: temporal
      factors: F=0 P=0 Q=0
      permutation: # MFPQ
    - name: global_buffer
      type: spatial
      factors: N=8
      permutation: #CMRSPQN
    - name: global_buffer
      type: temporal
      factors: R=0 S=0 T=0
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
      bypass: [Weights, Inputs]"""

    return mapspace


def _get_weight_stationary_yaml(): #filter_size, num_channels, max_filter_size):

    # p = max_filter_size - filter_size + 1
    # q = max_filter_size - filter_size + 1

    mapspace = f"""
mapspace:
  targets:
    - name: DRAM
      type: bypass
      keep: [Inputs, Outputs, Weights]
      bypass: []
    - name: DRAM
      type: temporal
      factors: M=0 R=0 S=0 T=0
      permutation: #MFPQ #RSTQPFCMN
    - name: global_buffer
      type: spatial
      factors: N=8
      permutation: #CMRSPQN
    - name: global_buffer
      type: temporal
      factors: P=0 Q=0
    # - name: scratchpad
    #   type: bypass
    #   keep: [Weights]
    #   bypass: [Inputs, Outputs]
    # - name: scratchpad
    #   type: temporal
    #   factors: F=3 P=3 Q=3
    #   permutation: #QPNCMSR
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


def _get_layer(filter_size, num_channels, max_filter_size):

    pad_channels = math.ceil(num_channels/3)*3
    pad_x = math.ceil(max_filter_size/3)*3
    pad_y = math.ceil(max_filter_size/3)*3

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

  instance:
    N: 8 # batch size
    C: 1 # in channels (number of kernels)
    M: 20 # out channels (number of kernels)
    R: {filter_size} # filter width
    S: {filter_size} # filter height
    T: {filter_size} # filter depth
    Q: {pad_x-filter_size+1} # output image width (x)
    P: {pad_y-filter_size+1} # output image height (y)
    F: {pad_channels-filter_size+1} # output image depth (z)
    Wstride: 1
    Hstride: 1
    Dstride: 1"""

    return layer_spec
