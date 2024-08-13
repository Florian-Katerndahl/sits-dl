from typing import Any, Optional
import numpy as np
import torch
from sits_dl.tensordatacube import Models, TensorDataCube as TDC

class Predict:
    def __init__(self, model: torch._dynamo.NNOptimizedModule, model_type: Models, dc: torch.Tensor, mask: Optional[np.ndarray], c_step: int, r_step: int, batch_size: int, device: torch.device, workers: int, *args, **kwargs) -> None:
        self.model: torch._dynamo.NNOptimizedModule = model
        self.model_type: Models = model_type
        self.dc: torch.Tensor = dc
        self.mask: Optional[np.ndarray] = mask
        self.c_step: int = c_step
        self.r_step: int = r_step
        self.batch_size: int = batch_size
        self.device: torch.device = device
        self.workers: int = workers


    # FIXME currently on supports Chris' model!
    # FIXME uses both compiled model and AMP at the moment
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        OUTPUT_DTYPE: torch.dtype = torch.float16 if self.model_type == Models.SBERT else torch.uint8
        AUTOCAST_DEVICE: str = "cuda" if "cuda" in self.device else "cpu"

        dl: torch.DataLoader = TDC.to_dataloader(self.dc, self.batch_size, workers=self.workers)
        prediction: torch.Tensor = torch.full((self.r_step * self.c_step,), fill_value=TDC.OUTPUT_NODATA, dtype=OUTPUT_DTYPE, device=self.device)

        with torch.autocast(device_type=AUTOCAST_DEVICE, dtype=OUTPUT_DTYPE), torch.inference_mode():
            if self.mask is not None:
                mask_torch: torch.Tensor = torch.from_numpy(self.mask).bool()
                for batch_index, batch in enumerate(dl):
                    for _, samples in enumerate(batch):
                        start: int = batch_index * self.batch_size
                        end: int = start + len(samples)
                        subset: torch.Tensor = mask_torch[start:end]
                        if not torch.any(subset):  # does skipping actually give a spped improvement? Or does it slow inference down since t(checking for null) > t(predicting null vector)
                            next
                        # ordering of subsetting and moving makes little to no difference time-wise but big difference memory-wise
                        input_tensor: torch.Tensor = samples[subset].to(self.device, non_blocking=True)
                        if self.model_type == Models.SBERT:
                            res = self.model(
                                    x=input_tensor[:,:,:-2],
                                    doy=input_tensor[:,:,-2].int(),
                                    mask=input_tensor[:,:,-1].int()).squeeze()
                        elif self.model_type == Models.TRANSFORMER:
                            raise NotImplementedError
                        else:
                            raise RuntimeError
                        prediction[start:end][subset] = res
            else:
                for batch_index, batch in enumerate(dl):
                    for _, samples in enumerate(batch):
                        start: int = batch_index * self.batch_size
                        end: int = start + len(samples)
                        samples = samples.to(self.device, non_blocking=True)
                        if self.model_type == Models.SBERT:
                            res = self.model(
                                    x=samples[:,:,:-2],
                                    doy=samples[:,:,-2].int(),
                                    mask=samples[:,:,-1].int()).squeeze()
                        elif self.model_type == Models.TRANSFORMER:
                            raise NotImplementedError
                        else:
                            raise RuntimeError
                        prediction[start:end] = res
        
        return torch.reshape(prediction, (self.r_step, self.c_step)).sigmoid().cpu()
