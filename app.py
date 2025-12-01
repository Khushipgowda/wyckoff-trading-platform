# app.py
import streamlit as st
import sys
from pathlib import Path
import base64

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Wyckoff Trading Intelligence Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Professional Wyckoff Trading Analysis Platform",
        "Report a bug": None,
        "Get help": None,
    },
)

# Session state init
if "current_page" not in st.session_state:
    st.session_state.current_page = "welcome"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Import workspace (IMPORTANT: from ui.layout, not ui.workspace)
from ui.layout import render_workspace


# ------------------------------------------------------------
# BACKGROUND IMAGE LOADER
# ------------------------------------------------------------
def get_background_image_css() -> str:
    image_path = Path(__file__).parent / "asserts" / "images" / "image1.png"

    if image_path.exists():
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        img_css = f"url('data:image/png;base64,{encoded}')"
    else:
        img_css = "none"

    css = f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        [data-testid="stSidebar"] {{display: none;}}

        .stApp {{
            background:
                linear-gradient(rgba(3,7,18,0.65), rgba(3,7,18,0.90)),
                {img_css} center center / cover no-repeat fixed;
            height: 100vh;
            overflow: hidden;
        }}

        .main .block-container {{
            padding: 0;
            margin: 0;
            max-width: 100%;
            height: 100vh;
            overflow: hidden;
        }}
        
        /* Floating particles background */
        .particles {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: 1;
        }}
        
        .particle {{
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(96, 165, 250, 0.5);
            border-radius: 50%;
            animation: float 20s infinite linear;
        }}
        
        @keyframes float {{
            from {{
                transform: translateY(100vh) translateX(0);
                opacity: 0;
            }}
            10% {{
                opacity: 1;
            }}
            90% {{
                opacity: 1;
            }}
            to {{
                transform: translateY(-100vh) translateX(100px);
                opacity: 0;
            }}
        }}

        /* Enhanced Title Design */
        .welcome-container {{
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            z-index: 100;
            width: 100%;
        }}
        
        .frame-corners {{
            position: fixed;
            width: 80%;
            max-width: 800px;
            height: 300px;
            bottom: 50px;
            left: 50%;
            transform: translateX(-50%);
            pointer-events: none;
            z-index: 99;
        }}
        
        .corner {{
            position: absolute;
            width: 60px;
            height: 60px;
            border: 2px solid rgba(96, 165, 250, 0.3);
        }}
        
        .corner-tl {{
            top: 0;
            left: 0;
            border-right: none;
            border-bottom: none;
            animation: pulse-corner 4s ease-in-out infinite;
        }}
        
        .corner-tr {{
            top: 0;
            right: 0;
            border-left: none;
            border-bottom: none;
            animation: pulse-corner 4s ease-in-out infinite 0.5s;
        }}
        
        .corner-bl {{
            bottom: 0;
            left: 0;
            border-right: none;
            border-top: none;
            animation: pulse-corner 4s ease-in-out infinite 1s;
        }}
        
        .corner-br {{
            bottom: 0;
            right: 0;
            border-left: none;
            border-top: none;
            animation: pulse-corner 4s ease-in-out infinite 1.5s;
        }}
        
        @keyframes pulse-corner {{
            0%, 100% {{
                opacity: 0.3;
                transform: scale(1);
            }}
            50% {{
                opacity: 0.8;
                transform: scale(1.1);
            }}
        }}
        
        .welcome-subtitle {{
            position: fixed;
            bottom: 200px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(90deg, #60a5fa, #93c5fd, #60a5fa);
            background-size: 200% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 0.85rem;
            letter-spacing: 0.3em;
            font-weight: 600;
            text-transform: uppercase;
            animation: gradient-shift 3s.ease-in-out infinite, fadeInUp 1.5s ease-out;
        }}
        
        @keyframes gradient-shift {{
            0%, 100% {{
                background-position: 0% 50%;
            }}
            50% {{
                background-position: 100% 50%;
            }}
        }}
        
        .welcome-title-chest {{
            position: fixed;
            bottom: 140px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            z-index: 100;
            color: #ffffff !important;
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            text-shadow: 
                0 0 20px rgba(96, 165, 250, 0.5),
                0 0 40px rgba(96, 165, 250, 0.3),
                0 4px 20px rgba(0,0,0,0.8);
            text-transform: uppercase;
            animation: fadeInUp 1.8s ease-out, text-glow 4s ease-in-out infinite;
        }}
        
        @keyframes text-glow {{
            0%, 100% {{
                filter: brightness(1);
            }}
            50% {{
                filter: brightness(1.1);
            }}
        }}
        
        .decorative-lines {{
            position: fixed;
            bottom: 125px;
            left: 50%;
            transform: translateX(-50%);
            width: 300px;
            height: 1px;
            background: linear-gradient(90deg, 
                transparent, 
                rgba(96, 165, 250, 0.6), 
                rgba(96, 165, 250, 0.6),
                transparent);
            z-index: 99;
            animation: expand-line 2s ease-out;
        }}
        
        @keyframes expand-line {{
            from {{
                width: 0;
                opacity: 0;
            }}
            to {{
                width: 300px;
                opacity: 1;
            }}
        }}
        
        .tagline {{
            position: fixed;
            bottom: 50px;
            left: 50%;
            transform: translateX(-50%);
            color: rgba(255,255,255,0.5);
            font-size: 0.75rem;
            letter-spacing: 0.15em;
            animation: fadeIn 3s ease-out;
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, 
                rgba(255,255,255,0.95), 
                rgba(255,255,255,0.85)) !important;
            color: #1e293b !important;
            border: none !important;
            padding: 0.7rem 2rem !important;
            border-radius: 50px !important;
            letter-spacing: 0.08em !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase !important;
            box-shadow: 
                0 0 20px rgba(96, 165, 250, 0.3),
                0 4px 15px rgba(0,0,0,0.2),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
            margin-top: 0 !important;
            animation: fadeIn 2.2s ease-out;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative;
            overflow: hidden;
        }}
        
        .stButton > button::before {{
            content: "";
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(96, 165, 250, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }}
        
        .stButton > button:hover {{
            transform: scale(1.02) !important;
            box-shadow: 
                0 0 30px rgba(96, 165, 250, 0.5),
                0 8px 25px rgba(0,0,0,0.3),
                inset 0 1px 0 rgba(255,255,255,1) !important;
            background: linear-gradient(135deg, 
                rgba(255,255,255,1), 
                rgba(255,255,255,0.95)) !important;
        }}
        
        .stButton > button:hover::before {{
            width: 300px;
            height: 300px;
        }}
        
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateX(-50%) translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }}
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        </style>
    """
    return css


# ------------------------------------------------------------
# WELCOME PAGE RENDER
# ------------------------------------------------------------
def render_welcome():
    st.markdown(get_background_image_css(), unsafe_allow_html=True)

    # Floating particles
    particles_html = """
    <div class="particles">
        <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
        <div class="particle" style="left: 20%; animation-delay: 2s;"></div>
        <div class="particle" style="left: 30%; animation-delay: 4s;"></div>
        <div class="particle" style="left: 40%; animation-delay: 6s;"></div>
        <div class="particle" style="left: 50%; animation-delay: 8s;"></div>
        <div class="particle" style="left: 60%; animation-delay: 10s;"></div>
        <div class="particle" style="left: 70%; animation-delay: 12s;"></div>
        <div class="particle" style="left: 80%; animation-delay: 14s;"></div>
        <div class="particle" style="left: 90%; animation-delay: 16s;"></div>
    </div>
    """

    frame_html = """
    <div class="frame-corners">
        <div class="corner corner-tl"></div>
        <div class="corner corner-tr"></div>
        <div class="corner corner-bl"></div>
        <div class="corner corner-br"></div>
    </div>
    """

    st.markdown(particles_html + frame_html, unsafe_allow_html=True)

    # Title and subtitle (no emojis)
    st.markdown(
        """
        <div class="welcome-subtitle">PROFESSIONAL TRADING INTELLIGENCE</div>
        <div class="welcome-title-chest">
            WYCKOFF TRADING PLATFORM
        </div>
        <div class="tagline">MASTER THE MARKET | TRADE WITH CONFIDENCE</div>
        """,
        unsafe_allow_html=True,
    )

    # Button positioning
    st.markdown(
        """
        <style>
            .stButton {
                position: fixed;
                bottom: 85px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 101;
                width: auto !important;
                max-width: 220px !important;
            }
            
            .stButton > button {
                width: auto !important;
                display: inline-block !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("GET STARTED", use_container_width=False):
            st.session_state.authenticated = True
            st.session_state.current_page = "workspace"
            st.rerun()


# ------------------------------------------------------------
# MAIN ROUTER
# ------------------------------------------------------------
def main():
    if not st.session_state.authenticated:
        render_welcome()
    else:
        render_workspace()


if __name__ == "__main__":
    main()
