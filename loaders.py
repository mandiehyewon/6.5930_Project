import os
from pathlib import Path
from yamlwidgets import YamlWidgets
from pytimeloop.app import ModelApp, MapperApp
from pytimeloop.accelergy_interface import invoke_accelergy
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

import logging, sys
logger = logging.getLogger('pytimeloop')
formatter = logging.Formatter(
    '[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ConfigRegistry:
    #################################
    # Single PE
    #################################
    SINGLE_PE_DIR                     = Path('designs/singlePE')
    SINGLE_PE_ARCH                    = SINGLE_PE_DIR / 'arch/single_PE_arch.yaml'
    SINGLE_PE_COMPONENTS_DIR          = SINGLE_PE_DIR / 'arch/components'
    SINGLE_PE_COMPONENT_SMART_STORAGE = SINGLE_PE_COMPONENTS_DIR / 'smart_storage.yaml'
    SINGLE_PE_COMPONENT_MAC_COMPUTE   = SINGLE_PE_COMPONENTS_DIR / 'mac_compute.yaml'
    SINGLE_PE_COMPONENT_REG_STORAGE   = SINGLE_PE_COMPONENTS_DIR / 'reg_storage.yaml'

    SINGLE_PE_MAP = Path('designs/singlePE/map/map.yaml')

    SINGLE_PE_AG_DIR                            = Path('designs/singlePE_ag')
    SINGLE_PE_AG_ARCH                           = SINGLE_PE_AG_DIR / 'arch/single_PE_arch.yaml'
    SINGLE_PE_AG_COMPONENTS_DIR                 = SINGLE_PE_AG_DIR / 'arch/components'
    SINGLE_PE_AG_COMPONENT_SMART_STORAGE        = SINGLE_PE_AG_COMPONENTS_DIR / 'smart_storage.yaml'
    SINGLE_PE_AG_COMPONENT_SMART_STORAGE_WIDGET = SINGLE_PE_AG_COMPONENTS_DIR / 'smart_storage-widget.yaml'
    SINGLE_PE_AG_COMPONENT_MAC_COMPUTE          = SINGLE_PE_AG_COMPONENTS_DIR / 'mac_compute.yaml'
    SINGLE_PE_AG_COMPONENT_REG_STORAGE          = SINGLE_PE_AG_COMPONENTS_DIR / 'reg_storage.yaml'

    SINGLE_PE_OS_DIR            = Path('designs/singlePE_os')
    SINGLE_PE_OS_ARCH           = SINGLE_PE_OS_DIR / 'arch/single_PE_arch.yaml'
    SINGLE_PE_OS_COMPONENTS_DIR = SINGLE_PE_OS_DIR / 'arch/components'

    SINGLE_PE_OS_MAP          = SINGLE_PE_OS_DIR / 'map/map.yaml'
    SINGLE_PE_OS_MAP_TEMPLATE = SINGLE_PE_OS_DIR / 'map/map-template.yaml'
    SINGLE_PE_OS_MAP_WIDGET   = SINGLE_PE_OS_DIR / 'map/map-widget.yaml'

    #################################
    # System Manual
    #################################
    SYSTEM_MANUAL_DIR      = Path('designs/system_manual')
    SYSTEM_1x16_ARCH       = SYSTEM_MANUAL_DIR / 'arch/system_arch_1x16.yaml'
    SYSTEM_2x8_ARCH_WIDGET = SYSTEM_MANUAL_DIR / 'arch/system_arch_2x8-widget.yaml'
    SYSTEM_4x4_ARCH_WIDGET = SYSTEM_MANUAL_DIR / 'arch/system_arch_4x4-widget.yaml'
    SYSTEM_COMPONENTS_DIR  = SYSTEM_MANUAL_DIR / 'arch/components'

    SYSTEM_MANUAL_MAP_WIDGET   = SYSTEM_MANUAL_DIR / 'map/example_mapping_for_1x16-widget.yaml'
    SYSTEM_MANUAL_MAP          = SYSTEM_MANUAL_DIR / 'map/example_mapping_for_1x16.yaml'
    SYSTEM_MANUAL_MAP_TEMPLATE = SYSTEM_MANUAL_DIR / 'map/template_map.yaml'

    #################################
    # System Auto
    #################################
    SYSTEM_AUTO_DIR     = Path('designs/system_auto')
    CONSTRAINTS_DIR     = SYSTEM_AUTO_DIR / 'constraints'
    EXAMPLE_CONSTRAINTS = CONSTRAINTS_DIR / 'example_constraints.yaml'
    RELAXED_CONSTRAINTS = CONSTRAINTS_DIR / 'relaxed_constraints.yaml'
    MAPPER              = SYSTEM_AUTO_DIR / 'mapper/mapper.yaml'

    #################################
    # PIM
    #################################
    PIM_DIR            = Path('designs/PIM')
    PIM_ARCH           = PIM_DIR / 'arch/system_PIM.yaml'
    PIM_COMPONENTS_DIR = PIM_DIR / 'arch/components'
    PIM_CONSTRAINTS    = PIM_DIR / 'constraints/constraints.yaml'
    PIM_MAPPER         = PIM_DIR / 'mapper/mapper.yaml'

    #################################
    # PIM Large
    #################################
    PIM_LARGE_DIR            = Path('designs/PIM_large')
    PIM_LARGE_ARCH           = PIM_LARGE_DIR / 'arch/system_PIM.yaml'
    PIM_LARGE_COMPONENTS_DIR = PIM_LARGE_DIR / 'arch/components'
    PIM_LARGE_CONSTRAINTS    = PIM_LARGE_DIR / 'constraints/constraints.yaml'
    PIM_LARGE_MAPPER         = PIM_LARGE_DIR / 'mapper/mapper.yaml'

    #################################
    # Problems
    #################################
    LAYER_SHAPES_DIR = Path('layer_shapes')
    TINY_LAYER_PROB  = LAYER_SHAPES_DIR / 'tiny_layer.yaml'
    SMALL_LAYER_PROB = LAYER_SHAPES_DIR / 'small_layer.yaml'
    MED_LAYER_PROB   = LAYER_SHAPES_DIR / 'medium_layer.yaml'
    DEPTHWISE_LAYER_PROB = LAYER_SHAPES_DIR / 'depth_wise.yaml'
    POINTWISE_LAYER_PROB = LAYER_SHAPES_DIR / 'point_wise.yaml'

def load_config(*paths):
    yaml = YAML(typ='safe')
    yaml.version = (1, 2)
    total = None
    def _collect_yaml(yaml_str, total):
        new_stuff = yaml.load(yaml_str)
        if total is None:
            return new_stuff

        for key, value in new_stuff.items():
            if key == 'compound_components' and key in total:
                total['compound_components']['classes'] += value['classes']
            elif key in total:
                raise RuntimeError(f'overlapping key: {key}')
            else:
                total[key] = value
        return total

    for path in paths:
        if isinstance(path, str):
            total = _collect_yaml(path, total)
            continue
        elif path.is_dir():
            for p in path.glob('*.yaml'):
                with p.open() as f:
                    total = _collect_yaml(f.read(), total)
        else:
            with path.open() as f:
                total = _collect_yaml(f.read(), total)
    return total

def load_config_str(*paths):
    total = ''
    for path in paths:
        if isinstance(path, str):
            return path
        elif path.is_dir():
            for p in path.glob('*.yaml'):
                with p.open() as f:
                    total += f.read() + '\n'
            return total
        else:
            with path.open() as f:
                return f.read()

def get_paths(paths):
    all_paths = []
    for path in paths:
        if path.is_dir():
            for p in path.glob('*.yaml'):
                all_paths.append(str(p))
        else:
            all_paths.append(str(path))
    return all_paths

def dump_str(yaml_dict):
    yaml = YAML(typ='safe')
    yaml.version = (1, 2)
    yaml.default_flow_style = False
    stream = StringIO()
    yaml.dump(yaml_dict, stream)
    return stream.getvalue()

def show_config(*paths):
    print(load_config_str(*paths))

def load_widget_config(*paths, title=''):
    widget = YamlWidgets(load_config_str(*paths), title=title)
    widget.display()
    return widget

def run_timeloop_model(*paths):
    yaml_str = dump_str(load_config(*paths))
    model = ModelApp(yaml_str, '.')
    result = model.run_subprocess()
    return result

def run_accelergy(*paths):
    has_str = False
    for path in paths:
        if isinstance(path, str):
            has_str = True
            break
    if has_str:
        yaml_str = dump_str(load_config(*paths))
        with open('tmp-accelergy.yaml', 'w') as f:
            f.write(yaml_str)
        result = invoke_accelergy(['tmp-accelergy.yaml'], '', '.')
        os.remove('tmp-accelergy.yaml')
    else:
        result = invoke_accelergy(get_paths(paths), '', '.')
    return result

def configure_mapping(config_str, mapping, var):
    config = load_config(config_str)
    for key in var:
        for level in config['options']:
            for level_key in level:
                if key == level_key:
                    var[key] = level[level_key]
    with open(mapping, 'r') as f:
        complete_mapping = eval(f.read())
    return complete_mapping

def run_timeloop_mapper(*paths):
    yaml_str = dump_str(load_config(*paths))
    mapper = MapperApp(yaml_str, '.')
    result = mapper.run_subprocess()
    return result