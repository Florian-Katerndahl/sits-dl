from typing import Optional, Any
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sits_dl.tensordatacube import TensorDataCube as TDC, Models, dataloader_from_chunk
from sits_dl.sbert import SBERTClassification

@torch.inference_mode()
def predict(model, data: torch.tensor, it: Models) -> Any:
    """
    Apply previously trained model to new data
    :param model: previously trained model
    :param torch.tensor data: new input data
    :return Any: Array of predictions
    """
    outputs = model(data)
    _, predicted = torch.max(outputs.data if it == Models.LSTM else outputs, 1)
    return predicted


def predict_lstm(lstm: torch.nn.LSTM, dc: torch.Tensor, mask: Optional[np.ndarray], c_step: int, r_step: int) -> torch.tensor:
    prediction: torch.Tensor = torch.full((r_step * c_step,), fill_value=TDC.OUTPUT_NODATA, dtype=torch.long)
    if mask:
        merged_row: torch.Tensor = torch.full(c_step, fill_value=TDC.OUTPUT_NODATA, dtype=torch.long)
        for chunk_rows in range(0, r_step):
            merged_row.zero_()
            squeezed_row: torch.Tensor = predict(
                lstm,
                dc[chunk_rows, mask[chunk_rows]],
                Models.LSTM)
            merged_row[mask[chunk_rows]] = squeezed_row
            prediction[chunk_rows, 0:c_step] = merged_row
    else:
        for chunk_rows in range(0, r_step):
            prediction[chunk_rows, 0:c_step] = predict(lstm, dc[chunk_rows], Models.LSTM)
    
    return prediction


@torch.inference_mode()
def predict_transformer(transformer: torch.nn.Transformer, dc: torch.Tensor, mask: Optional[np.ndarray], c_step: int, r_step: int, batch_size: int, device: torch.device) -> torch.Tensor:
    dl: DataLoader = dataloader_from_chunk(dc, batch_size)
    prediction: torch.Tensor = torch.full((r_step * c_step,), fill_value=TDC.OUTPUT_NODATA, dtype=torch.long)

    if mask is not None:
        mask_long: torch.Tensor = torch.from_numpy(np.reshape(mask, (-1,))).bool()
        for batch_index, batch in enumerate(dl):
            for _, samples in enumerate(batch):
                start: int = batch_index * batch_size
                end: int = start + len(samples)
                subset: torch.Tensor = mask_long[start:end]
                if not torch.any(subset):
                    next
                input_tensor: torch.Tensor = samples[subset].to(device, non_blocking=True)  # ordering of subsetting and moving makes little to no difference time-wise but big difference memory-wise
                prediction[start:end][subset] = predict(transformer, input_tensor, Models.TRANSFORMER).cpu()
    else:
        for batch_index, batch in enumerate(dl):
            for _, samples in enumerate(batch):
                start: int = batch_index * batch_size
                end: int = start + len(samples)
                prediction[start:end] = predict(transformer, samples.to(device, non_blocking=True), Models.TRANSFORMER).cpu()

    return torch.reshape(prediction, (r_step, c_step))


@torch.inference_mode()
def predict_sbert(sbert: SBERTClassification, dc: torch.Tensor, mask: Optional[np.ndarray], c_step: int, r_step: int, batch_size: int, device: torch.device) -> torch.Tensor:
    dl: DataLoader = dataloader_from_chunk(dc, batch_size)
    prediction: torch.Tensor = torch.full((r_step * c_step,), fill_value=TDC.OUTPUT_NODATA, dtype=torch.float)

    if mask is not None:
        raise NotImplementedError
    else:
        for batch_index, batch in enumerate(dl):
            for _, samples in enumerate(batch):
                start: int = batch_index * batch_size
                end: int = start + len(samples)
                samples = samples.to(device, non_blocking=True)
                res = torch.sigmoid(
                    sbert(x=samples[:,:,:-2],
                          doy=samples[:,:,-2].long(), 
                          mask=samples[:,:,-1].long())
                          ).cpu().squeeze()
                prediction[start:end] = res

    return torch.reshape(prediction, (r_step, c_step))

    
    pass
