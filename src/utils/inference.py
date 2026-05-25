import torch
import torch.nn.functional as F

def apply_model_to_spec(model, mixture_spec, segment_frames, overlap=0.25):
    """
    Applies the model to a large spectrogram using a sliding window.
    mixture_spec: (Channels, Freqs, Frames) - Complex tensor
    segment_frames: Number of frames for each chunk (e.g., matching training duration)
    overlap: Overlap ratio (0 to 1).
    """
    device = mixture_spec.device
    channels, freqs, total_frames = mixture_spec.shape
    
    # If the track is shorter than or equal to segment_frames, just run it once
    if total_frames <= segment_frames:
        with torch.no_grad():
            return model(mixture_spec.unsqueeze(0)).squeeze(0)

    hop = int(segment_frames * (1 - overlap))
    if hop <= 0:
        hop = 1
        
    # Output buffer (complex) and weight buffer
    output = torch.zeros((channels, freqs, total_frames), dtype=mixture_spec.dtype, device=device)
    weights = torch.zeros(total_frames, device=device)
    
    # Window for smooth blending
    window = torch.ones(segment_frames, device=device)
    if overlap > 0:
        fade_size = int(segment_frames * overlap)
        if fade_size > 0:
            fade_in = torch.linspace(0, 1, fade_size, device=device)
            fade_out = torch.linspace(1, 0, fade_size, device=device)
            window[:fade_size] *= fade_in
            window[-fade_size:] *= fade_out

    for start in range(0, total_frames - segment_frames + hop, hop):
        end = start + segment_frames
        
        # Handle the last chunk if it would go out of bounds
        if end > total_frames:
            end = total_frames
            start = end - segment_frames
        
        chunk = mixture_spec[:, :, start:end].unsqueeze(0)
        with torch.no_grad():
            estimate = model(chunk).squeeze(0)
            
        output[:, :, start:end] += estimate * window
        weights[start:end] += window
        
        if end == total_frames:
            break
        
    # Normalize by weights to handle overlap
    output /= weights.view(1, 1, -1).clamp(min=1e-7)
    
    return output
