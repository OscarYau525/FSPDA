# -*- coding: utf-8 -*-
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
import torch

from pcode.optim.sgd import SGD

from pcode.optim.dgc import DGC
from pcode.optim.parallel_choco import ParallelCHOCO
from pcode.optim.parallel_choco_v import ParallelCHOCO_V
from pcode.optim.ef_sign_sgd import EF_SignSGD
from pcode.optim.dcd_psgd import DCD_PSGD
from pcode.optim.ecd_psgd import ECD_PSGD
from pcode.optim.deep_squeeze import DeepSqueeze
from pcode.optim.gnsd import GNSD
from pcode.optim.detag import DeTAG
from pcode.optim.gt_hsgd import GtHsgd
from pcode.optim.parallel_docom_v import ParallelDoCoM_V
from pcode.optim.parallel_beer_v import ParallelBEER_V
from pcode.optim.fspda import FSPDA
from pcode.optim.fspda_storm import FSPDA_STORM
from pcode.optim.fspda_adam import FSPDA_ADAM
from pcode.optim.fspda_async import FSPDA_Async
from pcode.optim.fsppd_ef import FSPPD_EF
from pcode.optim.swarm_sgd import SwarmSGD
from pcode.optim.di_cs import Di_CS
from pcode.optim.di_cs_gt import Di_CS_GT
from pcode.optim.diging import DIGing
from pcode.optim.sparq_sgd import SPARQ_SGD
from pcode.optim.cp_sgd import CP_SGD
from pcode.optim.ticopd import TiCoPD
from pcode.optim.dimix import DIMIX
from pcode.optim.dsgd import DSGD
from pcode.optim.def_atc import DEF_ATC
from pcode.optim.gradient_tracking import GT
from pcode.optim.prox_skip import Prox_Skip
from pcode.optim.decen_scaffnew import Decen_Scaffnew
from pcode.optim.k_gt import K_GT
from pcode.optim.led import LED
from pcode.optim.dsgd_tv import DSGD_TV


def define_optimizer(conf, model):
    # define the param to optimize.
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": conf.weight_decay if "bn" not in key else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]

    # define the optimizer.
    opt_dict = {
        "sgd": SGD,
        "dgc": DGC,
        "dcd_psgd": DCD_PSGD,
        "ecd_psgd": ECD_PSGD,
        "parallel_choco": ParallelCHOCO,
        "parallel_choco_v": ParallelCHOCO_V,
        "ef_sign_sgd": EF_SignSGD,
        "deep_squeeze": DeepSqueeze,
        "gnsd": GNSD,
        "detag": DeTAG,
        "gt_hsgd": GtHsgd,
        "docom_v": ParallelDoCoM_V,
        "beer_v": ParallelBEER_V,
        "fspda": FSPDA,
        "fspda_storm": FSPDA_STORM,
        "fspda_adam": FSPDA_ADAM,
        "fspda_async": FSPDA_Async,
        "fsppd_ef": FSPPD_EF,
        "di_cs": Di_CS,
        "di_cs_gt": Di_CS_GT,
        "diging": DIGing,
        "swarm_sgd": SwarmSGD,
        "sparq_sgd": SPARQ_SGD,
        "cp_sgd": CP_SGD,
        "ticopd": TiCoPD,
        "dimix": DIMIX,
        "dsgd": DSGD,
        "def_atc": DEF_ATC,
        "gt": GT,
        "prox_skip": Prox_Skip,
        "decen_scaffnew": Decen_Scaffnew,
        "k_gt": K_GT,
        "led": LED,
        "dsgd_tv": DSGD_TV,
    }
    if conf.optimizer in opt_dict:
        optim_class = opt_dict[conf.optimizer]
    else:
        raise NotImplementedError

    return optim_class(
        params,
        lr=conf.lr,
        momentum=conf.momentum_factor,
        nesterov=conf.use_nesterov,
        conf=conf,
        model=model
    )
