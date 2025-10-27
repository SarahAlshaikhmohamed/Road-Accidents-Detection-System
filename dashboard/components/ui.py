import pathlib
import streamlit as st
import streamlit_antd_components as sac
from streamlit_extras.switch_page_button import switch_page

def sidebar_css() -> None:
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"]{display:none;}
    header[data-testid="stHeader"]{height:12px;padding:0;box-shadow:none;background:transparent;}
    main .block-container{padding-top:.5rem;}
    [data-testid="stSidebar"]{
      background: var(--surface);
      border-right: 1px solid var(--border);
    }
    .rads-sb-wrap{padding:10px 8px 14px;}
    .rads-sb-head{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
    .rads-sb-logo{
      width:40px;height:40px;border-radius:10px;border:1px solid var(--border);
      background: linear-gradient(145deg, #004D61, #007A8A);
      display:flex;align-items:center;justify-content:center; font-weight:900; color:#fff;
    }
    .rads-sb-title{font-weight:900; letter-spacing:.2px; line-height:1.1;}
    .rads-sb-sub{margin-top:-2px; opacity:.75; font-size:.9rem;}
    .rads-sb-hr{height:1px;background:var(--border);border:0;margin:12px 0;}
    .rads-sb-btn{
      display:inline-block;width:100%;text-align:center;font-weight:800;
      padding:10px 12px;border-radius:12px;border:1px solid var(--border);
      background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
    }
    .rads-sb-about{font-size:.95rem;color:var(--text-muted); line-height:1.55;}
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """sidebar including logo, GitHub button, about, and top menu"""
    sidebar_css()

    with st.sidebar:
        st.markdown('<div class="rads-sb-wrap">', unsafe_allow_html=True)

        # logo & title
        logo_url = None  # placeholder for now
        st.markdown('<div class="rads-sb-head">', unsafe_allow_html=True)
        if logo_url:
            st.image(logo_url, width=40)
        else:
            st.markdown('<div class="rads-sb-logo">R</div>', unsafe_allow_html=True)
        st.markdown(
            '<div><div class="rads-sb-title">RADS</div>'
            '<div class="rads-sb-sub">Road Accident Detection System</div></div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="rads-sb-hr"></div>', unsafe_allow_html=True)

        # >> GitHub 
        st.markdown(
            '<a class="rads-sb-btn" href="https://github.com/SarahAlshaikhmohamed/Road-Accidents-Detection-System" target="_blank">GitHub</a>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="rads-sb-hr"></div>', unsafe_allow_html=True)

        # >> About
        st.markdown("About the project")
        st.markdown(
            "<div class='rads-sb-about'>"
            "RADS is a system for real time accident & pothole detection, incident reporting, "
            "and analytics. Built with Streamlit & FastAPI, it connects live streams, detection "
            "models, and an RAG assistant for querying reports."
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown('<div class="rads-sb-hr"></div>', unsafe_allow_html=True)

        # Navigation menu 
        choice = sac.menu(
            items=[
                sac.MenuItem('Home', icon='house-fill'),
                sac.MenuItem('Overview', icon='bar-chart-line-fill'),
                sac.MenuItem('Dashboard', icon='grid-3x3-gap-fill'),
                sac.MenuItem('Live Stream', icon='camera-video-fill'),
                sac.MenuItem('Chat', icon='chat-dots-fill'),
            ],
            index=0,
            color='#004D61'
        )

        if choice == 'Overview':
            switch_page('overview')
        elif choice == 'Dashboard':
            switch_page('accidents')
        elif choice == 'Live Stream':
            switch_page('live stream')
        elif choice == 'Chat':
            switch_page('chat')

        st.markdown("</div>", unsafe_allow_html=True)