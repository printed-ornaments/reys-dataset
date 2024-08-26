from torch.optim import Adam, RMSprop

__all__ = ['get_optimizers']

def get_optimizers(cfg, space):
    lr = 1e-4
    encoder_what_lr = 5e-5
    encoder_cat_lr = 5e-5
    decoder_lr = 5e-5

    
    #fg_optimizer = get_optimizer(cfg.train.solver.fg.optim, cfg.train.solver.fg.lr, space.fg_module.parameters())
    bg_optimizer = get_optimizer(cfg.train.solver.bg.optim, cfg.train.solver.bg.lr, space.bg_module.parameters())
    
    encoder_params = list(map(id, space.gmair_fg.object_encoder_what.parameters()))
    decoder_params = list(map(id, space.gmair_fg.object_decoder.parameters()))
    encoder_cat_params = list(map(id, space.gmair_fg.object_encoder_cat.parameters()))
    
    pre_params = encoder_params + decoder_params + encoder_cat_params
    
    other_params = filter(lambda p: id(p) not in pre_params, space.gmair_fg.parameters())
    
    params = [
      {"params": space.gmair_fg.object_encoder_what.parameters(), "lr": encoder_what_lr},
      {"params": space.gmair_fg.object_decoder.parameters(), "lr": decoder_lr},
      {"params": space.gmair_fg.object_encoder_cat.parameters(), "lr": encoder_cat_lr},
      {"params": other_params, "lr": lr},
    ]
    
    gmair_optim = Adam(params, lr=lr)

    return gmair_optim, bg_optimizer
    
def get_optimizer(name, lr, param):
    optim_class = {
        'Adam': Adam,
        'RMSprop': RMSprop
    }[name]
    
    return optim_class(param, lr=lr)

