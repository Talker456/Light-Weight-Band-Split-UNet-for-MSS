import streamlit as st
import torch
import torchaudio
import io
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root directory to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ui.engine import AudioSeparatorEngine

# 1. Page Configuration & Custom Styling
st.set_page_config(
    page_title="Light Weight Roformer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme and UI consistency
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2129; padding: 10px; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e5b4e; color: white; }
    .stAudio { margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Engine Initialization (Cached for performance)
@st.cache_resource
def get_engine():
    """Initializes and returns the AudioSeparatorEngine."""
    return AudioSeparatorEngine()

@st.cache_data(show_spinner=False)
def load_ref_audio(uploaded_file, target_sr):
    """Caches the loading and resampling process to avoid redundant processing."""
    if uploaded_file is None:
        return None
    audio, sr = torchaudio.load(uploaded_file)
    if sr != target_sr:
        # Note: Resampling is heavy, so we cache the result
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    return audio

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Engine Load Error: {e}")
    st.stop()

# 3. Utility Functions
def audio_to_bytes(audio_tensor, sample_rate):
    """Converts a torch audio tensor to playable bytes for the browser."""
    buf = io.BytesIO()
    # Ensure tensor is on CPU before saving to buffer
    audio_cpu = audio_tensor.detach().cpu()
    torchaudio.save(buf, audio_cpu, sample_rate, format="wav")
    return buf.getvalue()

import librosa
import librosa.display

def plot_spectrogram(audio_tensor, title, use_container_width=True):
    """Calculates and renders a spectrogram using librosa."""
    # Convert torch tensor to numpy and merge channels (mean)
    y = audio_tensor.detach().cpu().numpy()
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    
    # Calculate STFT using librosa with engine's parameters
    stft = librosa.stft(
        y, 
        n_fft=engine.audio_engine.n_fft, 
        hop_length=engine.audio_engine.hop_length, 
        win_length=engine.audio_engine.win_length
    )
    
    # Convert to dB relative to maximum (following the example)
    db_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('dark_background')
    
    # Render with log-frequency scale
    img = librosa.display.specshow(
        db_spec, 
        sr=engine.sample_rate, 
        hop_length=engine.audio_engine.hop_length,
        x_axis='time', 
        y_axis='log', 
        cmap='magma',
        ax=ax
    )
    
    ax.set_title(title, fontsize=10, color='#00ffcc')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Freq (Hz)", fontsize=8)
    
    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=use_container_width)

# 4. Sidebar - System Information & Status
with st.sidebar:
    st.title("System Status")
    st.info(f"Computing Device: {'CUDA 🚀' if torch.cuda.is_available() else 'CPU 🐌'}")
    st.write(f"Sample Rate: {engine.sample_rate} Hz")
    st.divider()
    st.markdown("### Model Checkpoints")
    for stem in engine.stems:
        status = "✅ Loaded" if engine.models.get(stem) else "❌ Missing"
        st.write(f"- **{stem.capitalize()}**: {status}")

# 5. Main Application Layout
st.title("🎵 Light Weight Roformer GUI")
st.caption("AI-powered low-quality audio source separation with real-time DSP effects")

tab1, tab2 = st.tabs(["[1] Standard Separation", "[2] On-the-fly & Evaluation"])

# --- Tab 1: Standard Separation ---
with tab1:
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.subheader("Input Mixture")
        uploaded_file = st.file_uploader("Upload Audio (WAV, FLAC, MP3)", type=["wav", "flac", "mp3"], key="main_uploader")
        
        if uploaded_file:
            audio, sr = torchaudio.load(uploaded_file)
            # Resample if the input SR doesn't match the model requirement
            if sr != engine.sample_rate:
                audio = torchaudio.transforms.Resample(sr, engine.sample_rate)(audio)
            
            st.audio(uploaded_file, format="audio/wav")
            
            if st.button("🚀 Start Separation", key="btn_sep"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_ui(stem, val):
                    """Callback function to update progress bar and status text."""
                    status_text.text(f"Processing {stem.upper()}... ({val}%)")
                    progress_bar.progress(val / 100)

                with st.spinner("Isolating stems using Roformer..."):
                    results = engine.separate_audio(audio, callback=update_ui)
                    st.session_state['sep_results'] = results
                    st.session_state['original_mix'] = audio
                
                status_text.text("Separation Complete! ✨")
                progress_bar.empty()

    with col_viz:
        if uploaded_file:
            plot_spectrogram(audio, "Input Mixture Spectrogram")
        else:
            st.info("Please upload an audio file to visualize the mixture.")

    if 'sep_results' in st.session_state:
        st.divider()
        st.subheader("Isolated Stems")
        
        # Display results in a 2-column grid
        res_cols = st.columns(2)
        for i, (stem, stem_audio) in enumerate(st.session_state['sep_results'].items()):
            with res_cols[i % 2]:
                with st.container():
                    st.markdown(f"#### {stem.upper()}")
                    st.audio(audio_to_bytes(stem_audio, engine.sample_rate), format="audio/wav")
                    plot_spectrogram(stem_audio, f"Estimate: {stem.capitalize()}")
                    
                    st.download_button(
                        label=f"Download {stem.capitalize()}.wav",
                        data=audio_to_bytes(stem_audio, engine.sample_rate),
                        file_name=f"separated_{stem}.wav",
                        mime="audio/wav"
                    )

# --- Tab 2: On-the-fly & Evaluation ---
with tab2:
    st.subheader("Reference Mixing & BSS Metrics")
    
    # Initialize session state for references
    if 'ref_audios_data' not in st.session_state:
        st.session_state['ref_audios_data'] = {}

    exp_cols = st.columns(4)
    eq_params = {}

    for i, stem in enumerate(engine.stems):
        with exp_cols[i]:
            st.markdown(f"**{stem.upper()}**")
            file = st.file_uploader(f"Load {stem.capitalize()} Ref", type=["wav", "flac", "mp3"], key=f"ref_in_{stem}")
            
            if file:
                # Use cached loading
                st.session_state['ref_audios_data'][stem] = load_ref_audio(file, engine.sample_rate)

            with st.expander("DSP Settings", expanded=False):
                eq_params[stem] = {
                    'gain': st.slider("Gain (dB)", -12, 12, 0, key=f"g_{stem}"),
                    'low': st.slider("Low (dB)", -12, 12, 0, key=f"l_{stem}"),
                    'mid': st.slider("Mid (dB)", -12, 12, 0, key=f"m_{stem}"),
                    'high': st.slider("High (dB)", -12, 12, 0, key=f"h_{stem}")
                }

    st.divider()
    
    if st.button("🔥 Apply Effects, Mix & Evaluate Performance", key="btn_eval"):
        if not st.session_state['ref_audios_data']:
            st.warning("Please upload at least one reference stem to perform the evaluation.")
        else:
            processed_refs = {}
            max_len = 0
            
            # 1. Apply Effects and create the mixture
            with st.spinner("Applying EQ and Gain effects..."):
                for stem, audio in st.session_state['ref_audios_data'].items():
                    if audio is None: continue
                    audio_np = audio.numpy()
                    processed = engine.apply_effects(audio_np, eq_params[stem])
                    processed_refs[stem] = torch.from_numpy(processed)
                    max_len = max(max_len, processed.shape[1])
                
                mix_audio = torch.zeros((2, max_len))
                for stem, audio in processed_refs.items():
                    if audio.shape[1] < max_len:
                        audio = torch.nn.functional.pad(audio, (0, max_len - audio.shape[1]))
                    mix_audio += audio
            
            # 2. Run Separation on the modified mixture
            with st.spinner("Processing mixture through model..."):
                results = engine.separate_audio(mix_audio)
            
            # 3. Calculate BSS Evaluation Metrics
            with st.spinner("Calculating BSS Metrics (SDR/SIR/SAR)..."):
                scores = engine.calculate_bss_eval(processed_refs, results)
            
            st.success("Analysis Complete!")
            
            # Performance Metric Table
            df_scores = pd.DataFrame(scores).T
            st.table(df_scores.style.highlight_max(axis=0, color='#1e5b4e').format("{:.2f} dB"))
            
            # Visual & Audio Comparison
            st.subheader("Reference vs Estimate Comparison")
            for stem in engine.stems:
                if stem in processed_refs:
                    c1, c2 = st.columns(2)
                    with c1:
                        plot_spectrogram(processed_refs[stem], f"REFERENCE: {stem.upper()}")
                        st.audio(audio_to_bytes(processed_refs[stem], engine.sample_rate), format="audio/wav")
                    with c2:
                        plot_spectrogram(results[stem], f"ESTIMATE: {stem.upper()}")
                        st.audio(audio_to_bytes(results[stem], engine.sample_rate), format="audio/wav")

st.divider()
st.caption("Built with Streamlit & PyTorch | Light Weight Roformer Engine")
