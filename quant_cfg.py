from hqq.core.quantize import BaseQuantizeConfig

def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    
    for i in range(n_layers):
        if i < 4:
            q_config = BaseQuantizeConfig(nbits=8, group_size=64)
        else:
            q_config = BaseQuantizeConfig(nbits=4, group_size=64)
        
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q_config
        
    return quant_config