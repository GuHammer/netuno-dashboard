# -*- coding: utf-8 -*-
"""
Netuno Dados - Dashboard Premium
Performance Comercial - Grupo Netuno / R2F Capital

Tabs:
1. Performance Comercial (Visão Geral)
2. Proventos & Amortizações
3. Posição & Estratégia
4. PL & Análise Pareto
5. Saldo em Conta Corrente
6. Base de Clientes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
from datetime import datetime, timedelta

# ================== CONFIGURAÇÃO ==================
st.set_page_config(
    page_title="Netuno Dados | Performance Comercial",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cores da marca R2F / Netuno
CORES = {
    "primary": "#00acad",
    "primary_dark": "#008b8c",
    "secondary": "#1a2744",
    "accent": "#00acad",
    "positive": "#28a745",
    "negative": "#dc3545",
    "warning": "#ffc107",
    "neutral": "#6c757d",
    "bg_dark": "#1a2744",
    "bg_light": "#f8f9fa",
    "bg_card": "#ffffff",
    "border": "#e3e1e8",
    "text_dark": "#1a2744",
    "text_light": "#6c757d",
}

COLOR_SEQ = [
    "#00acad", "#1a2744", "#28a745", "#ffc107", "#dc3545",
    "#6c757d", "#17a2b8", "#6610f2", "#fd7e14", "#20c997",
    "#e83e8c", "#007bff", "#343a40", "#00796b", "#5c6bc0",
]

PLOTLY_LAYOUT = dict(
    font=dict(family="Arial, Helvetica, sans-serif", color=CORES["text_dark"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11)
    ),
)

# ================== CSS PREMIUM ==================
st.markdown(f"""
<style>
    /* Header */
    .main-header {{
        background: linear-gradient(135deg, {CORES['primary']} 0%, {CORES['secondary']} 100%);
        padding: 25px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
    }}
    .main-header h1 {{
        color: white;
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    .main-header p {{
        color: rgba(255,255,255,0.85);
        margin: 5px 0 0 0;
        font-size: 14px;
    }}

    /* KPI cards */
    .kpi-card {{
        background: white;
        padding: 16px 18px;
        border-radius: 10px;
        border-left: 4px solid {CORES['primary']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .kpi-card.accent {{
        border-left-color: {CORES['secondary']};
    }}
    .kpi-card.positive {{
        border-left-color: {CORES['positive']};
    }}
    .kpi-card.negative {{
        border-left-color: {CORES['negative']};
    }}
    .kpi-card.warning {{
        border-left-color: {CORES['warning']};
    }}
    .kpi-label {{
        font-size: 11px;
        color: {CORES['neutral']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 4px 0;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 22px;
        font-weight: 700;
        color: {CORES['secondary']};
        margin: 0;
        line-height: 1.2;
    }}
    .kpi-delta {{
        font-size: 12px;
        margin: 4px 0 0 0;
    }}
    .kpi-delta.pos {{ color: {CORES['positive']}; }}
    .kpi-delta.neg {{ color: {CORES['negative']}; }}
    .kpi-delta.neu {{ color: {CORES['neutral']}; }}

    /* Section headers */
    .section-header {{
        border-left: 4px solid {CORES['primary']};
        padding-left: 12px;
        margin: 20px 0 15px 0;
    }}
    .section-header h3 {{
        margin: 0;
        color: {CORES['secondary']};
        font-size: 16px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .section-header p {{
        margin: 2px 0 0 0;
        color: {CORES['neutral']};
        font-size: 12px;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: {CORES['bg_light']};
        padding: 4px;
        border-radius: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {CORES['primary']} !important;
        color: white !important;
    }}

    /* Info card */
    .info-card {{
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid {CORES['border']};
        margin-bottom: 12px;
    }}
    .info-card h4 {{
        margin: 0 0 8px 0;
        color: {CORES['secondary']};
        font-size: 14px;
    }}

    /* Highlight box */
    .highlight-box {{
        background: {CORES['secondary']};
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid {CORES['primary']};
    }}
    .highlight-box .hl-label {{
        color: {CORES['primary']};
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin: 0 0 6px 0;
    }}
    .highlight-box .hl-value {{
        color: white;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
    }}
    .highlight-box .hl-sub {{
        color: rgba(255,255,255,0.7);
        font-size: 12px;
        margin: 4px 0 0 0;
    }}

    /* Hide default elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Dataframe styling */
    .dataframe {{ font-size: 13px; }}

    /* Divider */
    .custom-divider {{
        border: 0;
        border-top: 1px solid {CORES['border']};
        margin: 20px 0;
    }}
</style>
""", unsafe_allow_html=True)


# ================== AUTENTICAÇÃO ==================
def check_password():
    """Verifica autenticação."""
    def password_entered():
        senha_correta = st.secrets.get("DASHBOARD_PASSWORD", "netuno2025")
        if st.session_state["password"] == senha_correta:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #00acad 0%, #1a2744 100%);
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            margin: 50px auto;
            max-width: 400px;
        ">
            <h1 style="color: white; margin-bottom: 10px;">🔱 NETUNO DADOS</h1>
            <p style="color: rgba(255,255,255,0.8);">Performance Comercial</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Senha de acesso:", type="password",
                          on_change=password_entered, key="password",
                          placeholder="Digite a senha...")
            st.markdown(
                "<p style='text-align: center; color: #6c757d; font-size: 12px;'>"
                "Grupo Netuno • R2F Capital</p>",
                unsafe_allow_html=True
            )
        return False

    elif not st.session_state["password_correct"]:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #00acad 0%, #1a2744 100%);
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            margin: 50px auto;
            max-width: 400px;
        ">
            <h1 style="color: white; margin-bottom: 10px;">🔱 NETUNO DADOS</h1>
            <p style="color: rgba(255,255,255,0.8);">Performance Comercial</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Senha de acesso:", type="password",
                          on_change=password_entered, key="password",
                          placeholder="Digite a senha...")
            st.error("Senha incorreta. Tente novamente.")
        return False

    return True


# ================== CONEXÃO SUPABASE ==================
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        st.error("Credenciais Supabase não configuradas!")
        st.info("Configure SUPABASE_URL e SUPABASE_KEY no Streamlit Cloud.")
        st.stop()
    return create_client(url, key)


def _fetch_all(table_name: str, order_col: str = None, desc: bool = False) -> pd.DataFrame:
    """Busca todos os registros de uma tabela, com paginação."""
    supabase = get_supabase_client()
    all_data = []
    batch = 1000
    offset = 0
    while True:
        q = supabase.table(table_name).select('*').range(offset, offset + batch - 1)
        if order_col:
            q = q.order(order_col, desc=desc)
        resp = q.execute()
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < batch:
            break
        offset += batch
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


# ================== CARREGAMENTO DE DADOS ==================
@st.cache_data(ttl=300)
def carregar_clientes() -> pd.DataFrame:
    return _fetch_all('clientes')


@st.cache_data(ttl=300)
def carregar_saldo_cc() -> pd.DataFrame:
    return _fetch_all('saldo_cc', order_col='data_referencia', desc=True)


@st.cache_data(ttl=300)
def carregar_pl_total() -> pd.DataFrame:
    return _fetch_all('pl_total', order_col='data_referencia', desc=True)


@st.cache_data(ttl=300)
def carregar_posicao() -> pd.DataFrame:
    return _fetch_all('posicao')


@st.cache_data(ttl=300)
def carregar_proventos() -> pd.DataFrame:
    return _fetch_all('proventos', order_col='data_pagamento')


# ================== HELPERS ==================
def has_col(df: pd.DataFrame, col: str) -> bool:
    """Verifica se coluna existe e tem dados."""
    return col in df.columns and not df[col].isna().all()


def safe_sum(df: pd.DataFrame, col: str) -> float:
    """Soma segura de uma coluna."""
    if has_col(df, col):
        return pd.to_numeric(df[col], errors='coerce').sum()
    return 0.0


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Converte coluna para numérico de forma segura."""
    if has_col(df, col):
        return pd.to_numeric(df[col], errors='coerce').fillna(0)
    return pd.Series([0] * len(df), index=df.index)


def formatar_moeda(valor) -> str:
    """Formata valor em R$."""
    if pd.isna(valor) or valor is None:
        return "R$ 0,00"
    try:
        v = float(valor)
        neg = v < 0
        s = f"R$ {abs(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"-{s}" if neg else s
    except (ValueError, TypeError):
        return "R$ 0,00"


def formatar_pct(valor) -> str:
    """Formata percentual."""
    if pd.isna(valor) or valor is None:
        return "0,00%"
    try:
        return f"{float(valor):.2f}%".replace(".", ",")
    except (ValueError, TypeError):
        return "0,00%"


def cor_valor(valor) -> str:
    """Retorna cor CSS baseada no valor."""
    try:
        v = float(valor)
        if v > 0:
            return CORES['positive']
        elif v < 0:
            return CORES['negative']
    except (ValueError, TypeError):
        pass
    return CORES['neutral']


def kpi_html(label: str, value: str, delta: str = None, delta_class: str = "neu",
             card_class: str = "") -> str:
    """Gera HTML de card KPI."""
    delta_html = ""
    if delta:
        delta_html = f'<p class="kpi-delta {delta_class}">{delta}</p>'
    return f"""
    <div class="kpi-card {card_class}">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{value}</p>
        {delta_html}
    </div>
    """


def section_header(title: str, subtitle: str = ""):
    """Renderiza cabeçalho de seção."""
    sub = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(f'<div class="section-header"><h3>{title}</h3>{sub}</div>',
                unsafe_allow_html=True)


def highlight_box(label: str, value: str, subtitle: str = ""):
    """Renderiza caixa destaque escura."""
    sub = f'<p class="hl-sub">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="highlight-box">
        <p class="hl-label">{label}</p>
        <p class="hl-value">{value}</p>
        {sub}
    </div>
    """, unsafe_allow_html=True)


def apply_plotly_style(fig, height=350):
    """Aplica estilo padrão aos gráficos Plotly."""
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=height,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=CORES['border'])
    fig.update_yaxes(showgrid=True, gridcolor=CORES['border'],
                     showline=False, zeroline=False)
    return fig


def filtrar_por_assessor(df: pd.DataFrame, filtro: str) -> pd.DataFrame:
    """Aplica filtro de assessor ao DataFrame."""
    if filtro and filtro != "Todos" and has_col(df, 'assessor'):
        return df[df['assessor'] == filtro].copy()
    return df.copy()


# ================== COMPONENTES ==================
def render_header():
    """Cabeçalho principal."""
    st.markdown("""
    <div class="main-header">
        <h1>🔱 NETUNO DADOS</h1>
        <p>Performance Comercial | Grupo Netuno • R2F Capital</p>
    </div>
    """, unsafe_allow_html=True)


def render_kpis_principais(df_clientes: pd.DataFrame):
    """Renderiza KPIs principais com percentuais."""
    if df_clientes.empty:
        st.warning("Nenhum dado de clientes encontrado.")
        return

    pl_total = safe_sum(df_clientes, 'pl_total')
    total_rf = safe_sum(df_clientes, 'renda_fixa')
    total_rv = safe_sum(df_clientes, 'renda_variavel')
    total_fundos = safe_sum(df_clientes, 'fundos')
    total_prev = safe_sum(df_clientes, 'previdencia')
    qtd_clientes = len(df_clientes)

    pct_rf = (total_rf / pl_total * 100) if pl_total > 0 else 0
    pct_rv = (total_rv / pl_total * 100) if pl_total > 0 else 0
    pct_fundos = (total_fundos / pl_total * 100) if pl_total > 0 else 0
    pct_prev = (total_prev / pl_total * 100) if pl_total > 0 else 0

    cols = st.columns(6)
    kpis = [
        ("Total Carteira", formatar_moeda(pl_total), f"{qtd_clientes} clientes", "neu", ""),
        ("Renda Fixa", formatar_moeda(total_rf), f"% PL RF {formatar_pct(pct_rf)}", "neu", "accent"),
        ("Renda Variavel", formatar_moeda(total_rv), f"% PL RV {formatar_pct(pct_rv)}", "neu", ""),
        ("Fundos", formatar_moeda(total_fundos), f"% PL FI {formatar_pct(pct_fundos)}", "neu", ""),
        ("Previdencia", formatar_moeda(total_prev), f"% PL Prev {formatar_pct(pct_prev)}", "neu", "accent"),
        ("Qtd Clientes", str(qtd_clientes),
         f"Ticket Medio: {formatar_moeda(pl_total / qtd_clientes if qtd_clientes > 0 else 0)}", "neu", ""),
    ]

    for col, (label, value, delta, delta_cls, card_cls) in zip(cols, kpis):
        with col:
            st.markdown(kpi_html(label, value, delta, delta_cls, card_cls),
                        unsafe_allow_html=True)


# ================== TAB 1: PERFORMANCE COMERCIAL ==================
def render_performance(df_clientes: pd.DataFrame, df_saldo: pd.DataFrame):
    """Tab Performance Comercial / Visão Geral."""
    if df_clientes.empty:
        st.info("Nenhum dado de clientes disponivel.")
        return

    # --- Ranking Top Clientes por PL ---
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_header("Ranking por PL Total", "Top clientes por patrimonio liquido")

        df_rank = df_clientes.nlargest(15, 'pl_total').copy()
        df_rank['rank'] = range(1, len(df_rank) + 1)
        df_rank['pl_fmt'] = df_rank['pl_total'].apply(formatar_moeda)

        # Gráfico horizontal
        fig = go.Figure(go.Bar(
            x=safe_numeric(df_rank, 'pl_total'),
            y=df_rank['nome'],
            orientation='h',
            text=df_rank['pl_fmt'],
            textposition='auto',
            textfont=dict(size=11),
            marker_color=CORES['primary'],
            marker_line=dict(width=0),
        ))
        fig.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                          xaxis=dict(visible=False))
        apply_plotly_style(fig, height=max(350, len(df_rank) * 28))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        section_header("Resumo por Assessor")

        if has_col(df_clientes, 'assessor'):
            df_assessor = df_clientes.groupby('assessor').agg(
                qtd=('conta', 'count') if has_col(df_clientes, 'conta') else ('nome', 'count'),
                pl_total=('pl_total', 'sum')
            ).reset_index().sort_values('pl_total', ascending=False)

            # Gráfico barras assessor
            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(
                x=df_assessor['assessor'],
                y=df_assessor['pl_total'],
                text=df_assessor['pl_total'].apply(formatar_moeda),
                textposition='outside',
                textfont=dict(size=10),
                marker_color=CORES['secondary'],
            ))
            apply_plotly_style(fig_a, height=280)
            fig_a.update_layout(
                xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=10, b=60),
            )
            st.plotly_chart(fig_a, use_container_width=True)

            # Tabela assessor
            df_a_display = df_assessor.copy()
            df_a_display['pl_total'] = df_a_display['pl_total'].apply(formatar_moeda)
            df_a_display.columns = ['Assessor', 'Qtd Clientes', 'PL Total']
            st.dataframe(df_a_display, use_container_width=True, hide_index=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Tabela Detalhada de Clientes ---
    section_header("Detalhamento por Cliente",
                   "Ranking completo com breakdown por classe de ativo")

    df_detail = df_clientes.sort_values('pl_total', ascending=False).copy()
    df_detail['rank'] = range(1, len(df_detail) + 1)

    cols_show = ['rank', 'nome']
    rename = {'rank': 'Rank', 'nome': 'Cliente'}

    if has_col(df_detail, 'assessor'):
        cols_show.append('assessor')
        rename['assessor'] = 'Assessor'

    for c, label in [('pl_total', 'PL Total'), ('renda_fixa', 'Renda Fixa'),
                     ('renda_variavel', 'Renda Variavel'), ('fundos', 'Fundos'),
                     ('previdencia', 'Previdencia')]:
        if has_col(df_detail, c):
            cols_show.append(c)
            rename[c] = label

    df_show = df_detail[cols_show].copy()
    for c in ['pl_total', 'renda_fixa', 'renda_variavel', 'fundos', 'previdencia']:
        if c in df_show.columns:
            df_show[c] = df_show[c].apply(formatar_moeda)
    df_show = df_show.rename(columns=rename)
    st.dataframe(df_show, use_container_width=True, hide_index=True, height=400)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Ranking Saldo CC ---
    if not df_saldo.empty and has_col(df_saldo, 'saldo'):
        col_s1, col_s2 = st.columns([2, 1])

        with col_s1:
            section_header("Rank por Saldo em Conta Corrente",
                           "Clientes com maior saldo parado")
            df_saldo_rank = df_saldo.nlargest(10, 'saldo')[['nome', 'saldo']].copy()
            df_saldo_rank['rank'] = range(1, len(df_saldo_rank) + 1)

            fig_s = go.Figure(go.Bar(
                x=safe_numeric(df_saldo_rank, 'saldo'),
                y=df_saldo_rank['nome'],
                orientation='h',
                text=df_saldo_rank['saldo'].apply(formatar_moeda),
                textposition='auto',
                textfont=dict(size=11),
                marker_color=CORES['warning'],
            ))
            fig_s.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                                xaxis=dict(visible=False))
            apply_plotly_style(fig_s, height=max(250, len(df_saldo_rank) * 28))
            fig_s.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_s, use_container_width=True)

        with col_s2:
            section_header("Resumo Saldo CC")
            total_saldo = df_saldo['saldo'].sum()
            highlight_box("Total em Conta Corrente", formatar_moeda(total_saldo),
                          f"{len(df_saldo)} clientes")

    # --- Share of Wallet ---
    if has_col(df_clientes, 'pl_total'):
        section_header("Share of Wallet", "Participacao de cada cliente no PL total")
        pl_total_geral = safe_sum(df_clientes, 'pl_total')
        if pl_total_geral > 0:
            df_share = df_clientes.nlargest(20, 'pl_total')[['nome', 'pl_total']].copy()
            df_share['share_pct'] = df_share['pl_total'] / pl_total_geral * 100

            fig_share = go.Figure(go.Bar(
                x=df_share['nome'],
                y=df_share['share_pct'],
                text=df_share['share_pct'].apply(lambda v: f"{v:.2f}%"),
                textposition='outside',
                textfont=dict(size=10),
                marker_color=CORES['primary'],
            ))
            apply_plotly_style(fig_share, height=320)
            fig_share.update_layout(
                xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
                yaxis=dict(title="% do PL Total", ticksuffix="%"),
                margin=dict(l=10, r=10, t=10, b=80),
            )
            st.plotly_chart(fig_share, use_container_width=True)


# ================== TAB 2: PROVENTOS & AMORTIZAÇÕES ==================
def render_proventos_premium(df_proventos: pd.DataFrame, df_clientes: pd.DataFrame):
    """Tab Proventos e Amortizações."""
    if df_proventos.empty:
        st.info("Nenhum dado de proventos encontrado.")
        return

    # Converte datas
    if has_col(df_proventos, 'data_pagamento'):
        df_proventos['data_pagamento'] = pd.to_datetime(
            df_proventos['data_pagamento'], errors='coerce')

    # Detecta coluna de tipo (Juros vs Amortização)
    tipo_col = None
    for c in ['tipo_evento', 'tipo', 'tipo_provento', 'evento']:
        if has_col(df_proventos, c):
            tipo_col = c
            break

    # Detecta coluna de valor
    valor_col = 'total_proventos'
    for c in ['total_proventos', 'valor', 'valor_bruto', 'valor_provento']:
        if has_col(df_proventos, c):
            valor_col = c
            break

    df_proventos[valor_col] = safe_numeric(df_proventos, valor_col)

    # Calcula totais por tipo se disponível
    has_tipo = tipo_col is not None
    if has_tipo:
        mask_juros = df_proventos[tipo_col].str.contains(
            'Juros|Cupom|Dividendo|JCP|Rendimento', case=False, na=False)
        mask_amort = df_proventos[tipo_col].str.contains(
            'Amortiza', case=False, na=False)
        total_juros = df_proventos.loc[mask_juros, valor_col].sum()
        total_amort = df_proventos.loc[mask_amort, valor_col].sum()
    else:
        total_juros = df_proventos[valor_col].sum()
        total_amort = 0

    total_geral = total_juros + total_amort

    # --- KPIs ---
    if has_tipo:
        cols_kpi = st.columns(4)
        kpis = [
            ("Valor Total Provento (Juros)", formatar_moeda(total_juros), "", "neu", ""),
            ("Valor Total Amortizacao", formatar_moeda(total_amort), "", "neu", "accent"),
            ("Totalizador", formatar_moeda(total_geral), "", "neu", "positive"),
            ("Qtd Ativos", str(df_proventos['ativo'].nunique() if has_col(df_proventos, 'ativo') else len(df_proventos)),
             "", "neu", "warning"),
        ]
    else:
        cols_kpi = st.columns(3)
        kpis = [
            ("Total Proventos", formatar_moeda(total_geral), "", "neu", ""),
            ("Qtd Registros", str(len(df_proventos)), "", "neu", "accent"),
            ("Qtd Ativos", str(df_proventos['ativo'].nunique() if has_col(df_proventos, 'ativo') else '-'),
             "", "neu", "warning"),
        ]

    for col, (label, value, delta, delta_cls, card_cls) in zip(cols_kpi, kpis):
        with col:
            st.markdown(kpi_html(label, value, delta, delta_cls, card_cls),
                        unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Tabela por Data ---
    if has_col(df_proventos, 'data_pagamento'):
        col1, col2 = st.columns([3, 2])

        with col1:
            section_header("Ativos totais e valor por data de pagamento")

            df_by_date = df_proventos.groupby(
                df_proventos['data_pagamento'].dt.date
            ).agg(
                qtd_ativos=(valor_col, 'count'),
                valor_total=(valor_col, 'sum'),
            ).reset_index()
            df_by_date = df_by_date.sort_values('data_pagamento')

            if has_tipo:
                df_juros_date = df_proventos[mask_juros].groupby(
                    df_proventos.loc[mask_juros, 'data_pagamento'].dt.date
                )[valor_col].sum().reset_index()
                df_juros_date.columns = ['data_pagamento', 'juros']

                df_amort_date = df_proventos[mask_amort].groupby(
                    df_proventos.loc[mask_amort, 'data_pagamento'].dt.date
                )[valor_col].sum().reset_index()
                df_amort_date.columns = ['data_pagamento', 'amortizacao']

                df_by_date = df_by_date.merge(df_juros_date, on='data_pagamento', how='left')
                df_by_date = df_by_date.merge(df_amort_date, on='data_pagamento', how='left')
                df_by_date[['juros', 'amortizacao']] = df_by_date[['juros', 'amortizacao']].fillna(0)

            # Gráfico dual axis
            fig_dates = make_subplots(specs=[[{"secondary_y": True}]])

            fig_dates.add_trace(go.Bar(
                x=df_by_date['data_pagamento'],
                y=df_by_date['valor_total'],
                name='Valor Total',
                marker_color=CORES['primary'],
                text=df_by_date['valor_total'].apply(formatar_moeda),
                textposition='outside',
                textfont=dict(size=9),
            ), secondary_y=False)

            fig_dates.add_trace(go.Scatter(
                x=df_by_date['data_pagamento'],
                y=df_by_date['qtd_ativos'],
                name='Ativos totais',
                mode='lines+markers+text',
                text=df_by_date['qtd_ativos'],
                textposition='top center',
                textfont=dict(size=10, color=CORES['secondary']),
                line=dict(color=CORES['secondary'], width=2),
                marker=dict(size=8),
            ), secondary_y=True)

            fig_dates.update_yaxes(title_text="Valor (R$)", secondary_y=False, showgrid=True,
                                   gridcolor=CORES['border'])
            fig_dates.update_yaxes(title_text="Qtd Ativos", secondary_y=True, showgrid=False)
            apply_plotly_style(fig_dates, height=350)
            fig_dates.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_dates, use_container_width=True)

        with col2:
            section_header("Qtd. clientes com proventos")

            # Clientes por data
            if has_col(df_proventos, 'conta'):
                df_cli_date = df_proventos.groupby(
                    df_proventos['data_pagamento'].dt.date
                )['conta'].nunique().reset_index()
                df_cli_date.columns = ['data_pagamento', 'qtd_clientes']
            elif has_col(df_proventos, 'nome'):
                df_cli_date = df_proventos.groupby(
                    df_proventos['data_pagamento'].dt.date
                )['nome'].nunique().reset_index()
                df_cli_date.columns = ['data_pagamento', 'qtd_clientes']
            else:
                df_cli_date = df_by_date[['data_pagamento', 'qtd_ativos']].rename(
                    columns={'qtd_ativos': 'qtd_clientes'})

            fig_cli = go.Figure(go.Scatter(
                x=df_cli_date['data_pagamento'],
                y=df_cli_date['qtd_clientes'],
                mode='lines+markers+text',
                text=df_cli_date['qtd_clientes'],
                textposition='top center',
                textfont=dict(size=11, color=CORES['primary']),
                line=dict(color=CORES['primary'], width=2.5),
                marker=dict(size=10, color=CORES['primary']),
                fill='tozeroy',
                fillcolor=f"rgba(0, 172, 173, 0.1)",
            ))
            apply_plotly_style(fig_cli, height=280)
            fig_cli.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cli, use_container_width=True)

            # Tabela resumo por data
            if has_tipo:
                df_tbl = df_by_date[['data_pagamento', 'juros', 'amortizacao', 'valor_total']].copy()
                for c in ['juros', 'amortizacao', 'valor_total']:
                    df_tbl[c] = df_tbl[c].apply(formatar_moeda)
                df_tbl.columns = ['Data', 'Proventos (Juros)', 'Amortizacao', 'Totalizador']
            else:
                df_tbl = df_by_date[['data_pagamento', 'valor_total', 'qtd_ativos']].copy()
                df_tbl['valor_total'] = df_tbl['valor_total'].apply(formatar_moeda)
                df_tbl.columns = ['Data', 'Valor Total', 'Qtd Ativos']
            st.dataframe(df_tbl, use_container_width=True, hide_index=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Breakdown por Ativo ---
    if has_col(df_proventos, 'ativo'):
        col_a, col_b = st.columns(2)

        with col_a:
            if has_tipo:
                section_header("Total Proventos por Ativo",
                               f"Total: {formatar_moeda(total_juros)}")
                df_juros_ativo = df_proventos[mask_juros].groupby('ativo')[valor_col].sum()
            else:
                section_header("Total por Ativo",
                               f"Total: {formatar_moeda(total_geral)}")
                df_juros_ativo = df_proventos.groupby('ativo')[valor_col].sum()

            df_juros_ativo = df_juros_ativo.nlargest(10).reset_index()
            df_juros_ativo.columns = ['Ativo', 'Valor']

            fig_at = go.Figure(go.Bar(
                x=df_juros_ativo['Valor'],
                y=df_juros_ativo['Ativo'],
                orientation='h',
                text=df_juros_ativo['Valor'].apply(formatar_moeda),
                textposition='auto',
                textfont=dict(size=10),
                marker_color=CORES['primary'],
            ))
            fig_at.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                                 xaxis=dict(visible=False))
            apply_plotly_style(fig_at, height=max(250, len(df_juros_ativo) * 30))
            fig_at.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_at, use_container_width=True)

        with col_b:
            if has_tipo:
                section_header("Total Amortizacao por Ativo",
                               f"Total: {formatar_moeda(total_amort)}")
                df_amort_ativo = df_proventos[mask_amort].groupby('ativo')[valor_col].sum()
                df_amort_ativo = df_amort_ativo.nlargest(10).reset_index()
                df_amort_ativo.columns = ['Ativo', 'Valor']

                fig_am = go.Figure(go.Bar(
                    x=df_amort_ativo['Valor'],
                    y=df_amort_ativo['Ativo'],
                    orientation='h',
                    text=df_amort_ativo['Valor'].apply(formatar_moeda),
                    textposition='auto',
                    textfont=dict(size=10),
                    marker_color=CORES['secondary'],
                ))
                fig_am.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                                     xaxis=dict(visible=False))
                apply_plotly_style(fig_am, height=max(250, len(df_amort_ativo) * 30))
                fig_am.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_am, use_container_width=True)
            else:
                # Distribuição por tipo de ativo (inferido do nome)
                section_header("Distribuicao por Tipo de Ativo")
                df_proventos['tipo_ativo_inf'] = df_proventos['ativo'].apply(_inferir_tipo_ativo)
                df_tipo = df_proventos.groupby('tipo_ativo_inf')[valor_col].sum().reset_index()
                df_tipo.columns = ['Tipo', 'Valor']
                df_tipo = df_tipo.sort_values('Valor', ascending=False)

                fig_tp = px.pie(df_tipo, values='Valor', names='Tipo',
                                color_discrete_sequence=COLOR_SEQ)
                apply_plotly_style(fig_tp, height=300)
                st.plotly_chart(fig_tp, use_container_width=True)

    # --- Tabela detalhada ---
    section_header("Detalhamento de Proventos")
    df_det = df_proventos.copy()
    cols_det = []
    rename_det = {}
    for c, label in [('nome', 'Cliente'), ('ativo', 'Ativo'),
                     (tipo_col, 'Tipo') if tipo_col else (None, None),
                     (valor_col, 'Valor'), ('data_pagamento', 'Data Pagamento')]:
        if c and has_col(df_det, c):
            cols_det.append(c)
            rename_det[c] = label
    if cols_det:
        df_det_show = df_det[cols_det].copy()
        if valor_col in df_det_show.columns:
            df_det_show[valor_col] = df_det_show[valor_col].apply(formatar_moeda)
        df_det_show = df_det_show.rename(columns=rename_det)
        st.dataframe(df_det_show, use_container_width=True, hide_index=True, height=400)


def _inferir_tipo_ativo(nome: str) -> str:
    """Infere tipo do ativo pelo nome."""
    if pd.isna(nome):
        return "Outros"
    nome = str(nome).upper()
    if any(t in nome for t in ['DEB', 'DEBENTURE']):
        return "Debenture"
    if 'CRA' in nome:
        return "CRA"
    if 'CRI' in nome:
        return "CRI"
    if any(t in nome for t in ['LCA', 'LCI']):
        return "LCA/LCI"
    if 'CDB' in nome:
        return "CDB"
    if 'FII' in nome or 'FUNDO IMOB' in nome:
        return "FII"
    if any(t in nome for t in ['TESOURO', 'LFT', 'LTN', 'NTN']):
        return "Tesouro"
    return "Outros"


# ================== TAB 3: POSIÇÃO & ESTRATÉGIA ==================
def render_posicao_estrategia(df_posicao: pd.DataFrame, df_clientes: pd.DataFrame,
                               filtro_assessor: str):
    """Tab Posição e Estratégia."""
    if df_posicao.empty:
        st.info("Nenhum dado de posicao encontrado.")
        return

    df = filtrar_por_assessor(df_posicao, filtro_assessor) if has_col(df_posicao, 'assessor') else df_posicao.copy()

    # Merge assessor se necessário
    if not has_col(df, 'assessor') and not df_clientes.empty and has_col(df_clientes, 'assessor'):
        df = df.merge(df_clientes[['conta', 'assessor']], on='conta', how='left')
        df = filtrar_por_assessor(df, filtro_assessor)

    # Detecta colunas de valor
    valor_col = 'valor_bruto'
    for c in ['valor_bruto', 'valor_total', 'valor']:
        if has_col(df, c):
            valor_col = c
            break
    df[valor_col] = safe_numeric(df, valor_col)

    # --- Distribuição por Mercado ---
    if has_col(df, 'mercado'):
        col1, col2 = st.columns([1, 1])

        with col1:
            section_header("Distribuicao por Tipo de Mercado")
            df_mercado = df.groupby('mercado').agg(
                valor=(valor_col, 'sum'),
                qtd_clientes=('conta', 'nunique') if has_col(df, 'conta') else (valor_col, 'count'),
            ).reset_index().sort_values('valor', ascending=False)

            total_pos = df_mercado['valor'].sum()
            df_mercado['pct'] = df_mercado['valor'] / total_pos * 100 if total_pos > 0 else 0

            fig_pie = px.pie(
                df_mercado, values='valor', names='mercado',
                color_discrete_sequence=COLOR_SEQ,
                hole=0.4,
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                                  textfont_size=11)
            apply_plotly_style(fig_pie, height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            section_header("Resumo por Mercado")
            df_m_show = df_mercado[['mercado', 'valor', 'pct', 'qtd_clientes']].copy()
            df_m_show['valor'] = df_m_show['valor'].apply(formatar_moeda)
            df_m_show['pct'] = df_m_show['pct'].apply(formatar_pct)
            df_m_show.columns = ['Mercado', 'Valor Total', '% do Total', 'Qtd Clientes']
            st.dataframe(df_m_show, use_container_width=True, hide_index=True)

            # Treemap
            section_header("Treemap de Posicoes")
            tree_cols = ['mercado']
            if has_col(df, 'ativo'):
                tree_cols.append('ativo')
            elif has_col(df, 'produto'):
                tree_cols.append('produto')

            if len(tree_cols) > 1:
                df_tree = df.groupby(tree_cols)[valor_col].sum().reset_index()
                fig_tree = px.treemap(
                    df_tree, path=tree_cols, values=valor_col,
                    color_discrete_sequence=COLOR_SEQ,
                )
                apply_plotly_style(fig_tree, height=350)
                st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Alocação por Estratégia (indexador) ---
    indexador_col = None
    for c in ['indexador', 'indice', 'estrategia', 'tipo_rendimento']:
        if has_col(df, c):
            indexador_col = c
            break

    if indexador_col:
        section_header("Alocacao por Estrategia/Indexador",
                       "Distribuicao do patrimonio por tipo de rentabilidade")

        col_e1, col_e2 = st.columns([2, 1])

        with col_e1:
            # Percentual por indexador por cliente
            df_idx = df.groupby([
                'nome' if has_col(df, 'nome') else 'conta',
                indexador_col
            ])[valor_col].sum().reset_index()

            nome_col = 'nome' if has_col(df, 'nome') else 'conta'
            df_pivot = df_idx.pivot_table(
                index=nome_col, columns=indexador_col, values=valor_col,
                aggfunc='sum', fill_value=0
            )
            df_pivot_pct = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

            # Top 15 clientes por total
            top_clientes = df_pivot.sum(axis=1).nlargest(15).index
            df_pivot_pct_show = df_pivot_pct.loc[top_clientes].copy()

            for c in df_pivot_pct_show.columns:
                df_pivot_pct_show[c] = df_pivot_pct_show[c].apply(formatar_pct)

            st.dataframe(df_pivot_pct_show, use_container_width=True)

        with col_e2:
            # Concentração de risco
            section_header("Concentracao de Risco",
                           "Qtd clientes com >=10% em cada estrategia")
            risk_data = []
            for idx_name in df_pivot_pct.columns:
                qtd = (df_pivot_pct[idx_name] >= 10).sum()
                risk_data.append({'Estrategia': idx_name, 'Qtd Clientes': qtd})
            df_risk = pd.DataFrame(risk_data).sort_values('Qtd Clientes', ascending=True)

            fig_risk = go.Figure(go.Bar(
                x=df_risk['Qtd Clientes'],
                y=df_risk['Estrategia'],
                orientation='h',
                text=df_risk['Qtd Clientes'],
                textposition='auto',
                marker_color=CORES['primary'],
            ))
            fig_risk.update_layout(yaxis=dict(tickfont=dict(size=11)),
                                   xaxis=dict(visible=False))
            apply_plotly_style(fig_risk, height=max(200, len(df_risk) * 30))
            fig_risk.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Vencimentos ---
    venc_col = None
    for c in ['vencimento', 'data_vencimento', 'dt_vencimento']:
        if has_col(df, c):
            venc_col = c
            break

    if venc_col:
        section_header("Calendario de Vencimentos",
                       "Ativos por data de vencimento")

        df['_venc'] = pd.to_datetime(df[venc_col], errors='coerce')
        df_venc = df.dropna(subset=['_venc']).copy()
        hoje = pd.Timestamp.now()

        # KPIs de vencimento
        vencidos = df_venc[df_venc['_venc'] < hoje]
        ate_d5 = df_venc[(df_venc['_venc'] >= hoje) & (df_venc['_venc'] <= hoje + timedelta(days=5))]
        acima_d5 = df_venc[df_venc['_venc'] > hoje + timedelta(days=5)]

        cols_v = st.columns(3)
        with cols_v[0]:
            st.markdown(kpi_html("Acima de D+5", str(len(acima_d5)),
                                 card_class="positive"), unsafe_allow_html=True)
        with cols_v[1]:
            st.markdown(kpi_html("Vencido", str(len(vencidos)),
                                 card_class="negative"), unsafe_allow_html=True)
        with cols_v[2]:
            st.markdown(kpi_html("Ate D+5", str(len(ate_d5)),
                                 card_class="warning"), unsafe_allow_html=True)

        # Tabela de vencimentos agrupada
        df_venc_grp = df_venc.groupby(df_venc['_venc'].dt.date).agg(
            valor_total=(valor_col, 'sum'),
            qtd=(valor_col, 'count'),
        ).reset_index().sort_values('_venc')
        df_venc_grp.columns = ['Data Vencimento', 'Total Valor Bruto', 'Qtd']
        df_venc_grp['Total Valor Bruto'] = df_venc_grp['Total Valor Bruto'].apply(formatar_moeda)
        st.dataframe(df_venc_grp, use_container_width=True, hide_index=True, height=300)

    # --- Top Ações Alocadas ---
    if has_col(df, 'mercado') and has_col(df, 'ativo'):
        df_rv = df[df['mercado'].str.contains('Vari|Acao|Ação|Stock|Equity', case=False, na=False)]
        if not df_rv.empty:
            section_header("Top Acoes Alocadas",
                           "Maiores posicoes em renda variavel")

            df_acoes = df_rv.groupby('ativo')[valor_col].sum().nlargest(15).reset_index()
            df_acoes.columns = ['Ativo', 'Valor']

            fig_ac = go.Figure(go.Bar(
                x=df_acoes['Valor'],
                y=df_acoes['Ativo'],
                orientation='h',
                text=df_acoes['Valor'].apply(formatar_moeda),
                textposition='auto',
                textfont=dict(size=10),
                marker_color=COLOR_SEQ[:len(df_acoes)],
            ))
            fig_ac.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                                 xaxis=dict(visible=False))
            apply_plotly_style(fig_ac, height=max(300, len(df_acoes) * 28))
            fig_ac.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_ac, use_container_width=True)


# ================== TAB 4: PL & ANÁLISE PARETO ==================
def render_pareto(df_clientes: pd.DataFrame, df_posicao: pd.DataFrame):
    """Tab PL e Análise Pareto."""
    if df_clientes.empty:
        st.info("Nenhum dado de clientes disponivel.")
        return

    # --- Análise Pareto ---
    section_header("Analise Pareto (Regra 80/20)",
                   "Concentracao do patrimonio por cliente")

    df_sorted = df_clientes.sort_values('pl_total', ascending=False).reset_index(drop=True)
    total_pl = df_sorted['pl_total'].sum()
    if total_pl <= 0:
        st.warning("PL total igual a zero.")
        return

    df_sorted['pct'] = df_sorted['pl_total'] / total_pl * 100
    df_sorted['pct_acumulado'] = df_sorted['pct'].cumsum()
    df_sorted['rank'] = range(1, len(df_sorted) + 1)

    # Encontra threshold 80%
    clientes_top_80 = (df_sorted['pct_acumulado'] <= 80).sum() + 1
    total_clientes = len(df_sorted)

    # KPIs Pareto
    cols_p = st.columns(3)
    with cols_p[0]:
        highlight_box("Clientes Top 80%", str(clientes_top_80),
                       f"de {total_clientes} clientes ({clientes_top_80/total_clientes*100:.0f}%)")
    with cols_p[1]:
        # Ágio/Deságio se disponível
        if not df_posicao.empty:
            valor_curva_col = None
            valor_mercado_col = None
            for c in ['valor_curva', 'pu_curva', 'preco_curva']:
                if has_col(df_posicao, c):
                    valor_curva_col = c
                    break
            for c in ['valor_mercado', 'pu_mercado', 'preco_mercado', 'valor_bruto']:
                if has_col(df_posicao, c):
                    valor_mercado_col = c
                    break

            if valor_curva_col and valor_mercado_col and valor_curva_col != valor_mercado_col:
                total_curva = safe_sum(df_posicao, valor_curva_col)
                total_merc = safe_sum(df_posicao, valor_mercado_col)
                agio = total_merc - total_curva
                if agio >= 0:
                    highlight_box("Valor Agio", formatar_moeda(agio),
                                   "Mercado > Curva")
                else:
                    highlight_box("Valor Desagio", formatar_moeda(agio),
                                   "Mercado < Curva")
            else:
                highlight_box("PL Total", formatar_moeda(total_pl))
        else:
            highlight_box("PL Total", formatar_moeda(total_pl))

    with cols_p[2]:
        ticket_medio = total_pl / total_clientes if total_clientes > 0 else 0
        highlight_box("Ticket Medio", formatar_moeda(ticket_medio),
                       f"PL / {total_clientes} clientes")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Gráfico Pareto ---
    section_header("Total PL e Pareto por Cliente")

    df_chart = df_sorted.head(30).copy()  # Top 30 para legibilidade

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])

    fig_pareto.add_trace(go.Bar(
        x=df_chart['nome'],
        y=df_chart['pl_total'],
        name='Total PL',
        marker_color=CORES['primary'],
        opacity=0.85,
    ), secondary_y=False)

    fig_pareto.add_trace(go.Scatter(
        x=df_chart['nome'],
        y=df_chart['pct_acumulado'],
        name='Pareto (%)',
        mode='lines+markers+text',
        text=df_chart['pct_acumulado'].apply(lambda v: f"{v:.1f}%"),
        textposition='top center',
        textfont=dict(size=9, color=CORES['secondary']),
        line=dict(color=CORES['secondary'], width=2.5),
        marker=dict(size=6, color=CORES['secondary']),
    ), secondary_y=True)

    # Linha 80%
    fig_pareto.add_hline(
        y=80, line_dash="dash", line_color=CORES['negative'],
        annotation_text="80%", secondary_y=True,
        annotation_font_color=CORES['negative'],
    )

    fig_pareto.update_yaxes(title_text="PL Total (R$)", secondary_y=False,
                             showgrid=True, gridcolor=CORES['border'])
    fig_pareto.update_yaxes(title_text="% Acumulado", secondary_y=True,
                             range=[0, 105], showgrid=False, ticksuffix="%")
    fig_pareto.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    apply_plotly_style(fig_pareto, height=420)
    fig_pareto.update_layout(margin=dict(l=10, r=10, t=40, b=80))
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Ranking Clientes (Pareto) - barra horizontal ---
    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        section_header("Rank Clientes (Pareto)", "Top 20 por participacao")
        df_top20 = df_sorted.head(20).copy()

        fig_rank = go.Figure(go.Bar(
            x=df_top20['pct'],
            y=df_top20['nome'],
            orientation='h',
            text=df_top20['pct'].apply(lambda v: f"{v:.2f}%"),
            textposition='auto',
            textfont=dict(size=10),
            marker_color=[CORES['primary'] if i < clientes_top_80 else CORES['neutral']
                          for i in range(len(df_top20))],
        ))
        fig_rank.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                               xaxis=dict(title="% do PL Total", ticksuffix="%"))
        apply_plotly_style(fig_rank, height=max(400, len(df_top20) * 24))
        fig_rank.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_rank, use_container_width=True)

    with col_r2:
        section_header("Tabela Pareto Completa")
        df_pareto_tbl = df_sorted[['rank', 'nome', 'pl_total', 'pct', 'pct_acumulado']].copy()
        df_pareto_tbl['pl_total'] = df_pareto_tbl['pl_total'].apply(formatar_moeda)
        df_pareto_tbl['pct'] = df_pareto_tbl['pct'].apply(formatar_pct)
        df_pareto_tbl['pct_acumulado'] = df_pareto_tbl['pct_acumulado'].apply(formatar_pct)
        df_pareto_tbl.columns = ['Rank', 'Cliente', 'PL Total', '% Individual', '% Acumulado']
        st.dataframe(df_pareto_tbl, use_container_width=True, hide_index=True, height=500)

    # --- Ágio/Deságio por Ativo (se disponível) ---
    if not df_posicao.empty:
        valor_curva_col = None
        valor_mercado_col = None
        for c in ['valor_curva', 'pu_curva']:
            if has_col(df_posicao, c):
                valor_curva_col = c
                break
        for c in ['valor_mercado', 'pu_mercado']:
            if has_col(df_posicao, c):
                valor_mercado_col = c
                break

        if valor_curva_col and valor_mercado_col:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            section_header("Agio/Desagio por Ativo",
                           "Diferenca entre valor de mercado e valor na curva")

            df_ad = df_posicao.copy()
            df_ad['_curva'] = safe_numeric(df_ad, valor_curva_col)
            df_ad['_mercado'] = safe_numeric(df_ad, valor_mercado_col)
            df_ad['_agio'] = df_ad['_mercado'] - df_ad['_curva']

            ativo_col = 'ativo' if has_col(df_ad, 'ativo') else 'produto' if has_col(df_ad, 'produto') else None

            if ativo_col:
                df_agio_grp = df_ad.groupby(ativo_col)['_agio'].sum().reset_index()
                df_agio_grp.columns = ['Ativo', 'Agio']

                col_ag, col_dg = st.columns(2)

                with col_ag:
                    section_header("Top Agio", "Ativos com maior valorizacao")
                    df_top_agio = df_agio_grp[df_agio_grp['Agio'] > 0].nlargest(10, 'Agio')
                    if not df_top_agio.empty:
                        fig_agio = go.Figure(go.Bar(
                            x=df_top_agio['Agio'],
                            y=df_top_agio['Ativo'],
                            orientation='h',
                            text=df_top_agio['Agio'].apply(formatar_moeda),
                            textposition='auto',
                            textfont=dict(size=10),
                            marker_color=CORES['positive'],
                        ))
                        fig_agio.update_layout(
                            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                            xaxis=dict(visible=False))
                        apply_plotly_style(fig_agio, height=max(200, len(df_top_agio) * 28))
                        fig_agio.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig_agio, use_container_width=True)

                with col_dg:
                    section_header("Top Desagio", "Ativos com maior desvalorizacao")
                    df_top_desagio = df_agio_grp[df_agio_grp['Agio'] < 0].nsmallest(10, 'Agio')
                    if not df_top_desagio.empty:
                        fig_desagio = go.Figure(go.Bar(
                            x=df_top_desagio['Agio'].abs(),
                            y=df_top_desagio['Ativo'],
                            orientation='h',
                            text=df_top_desagio['Agio'].apply(formatar_moeda),
                            textposition='auto',
                            textfont=dict(size=10),
                            marker_color=CORES['negative'],
                        ))
                        fig_desagio.update_layout(
                            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                            xaxis=dict(visible=False))
                        apply_plotly_style(fig_desagio, height=max(200, len(df_top_desagio) * 28))
                        fig_desagio.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig_desagio, use_container_width=True)


# ================== TAB 5: SALDO EM CONTA CORRENTE ==================
def render_saldo_cc_premium(df_saldo: pd.DataFrame, df_clientes: pd.DataFrame):
    """Tab Saldo em Conta Corrente."""
    if df_saldo.empty:
        st.info("Nenhum dado de saldo encontrado.")
        return

    df_saldo['saldo'] = safe_numeric(df_saldo, 'saldo')

    # Filtra saldo > 0
    df_pos = df_saldo[df_saldo['saldo'] > 0].copy()

    # Junta com assessor
    if not df_clientes.empty and has_col(df_clientes, 'assessor') and has_col(df_pos, 'conta'):
        df_pos = df_pos.merge(df_clientes[['conta', 'assessor']], on='conta', how='left')

    # --- KPIs ---
    total_saldo = df_pos['saldo'].sum()
    qtd_clientes = len(df_pos)
    saldo_medio = total_saldo / qtd_clientes if qtd_clientes > 0 else 0

    # Saldo > 500 (recurso ocioso significativo)
    df_alto = df_pos[df_pos['saldo'] > 500]
    total_alto = df_alto['saldo'].sum()

    cols_k = st.columns(4)
    with cols_k[0]:
        st.markdown(kpi_html("Total em CC", formatar_moeda(total_saldo),
                             f"{qtd_clientes} clientes", card_class="warning"),
                    unsafe_allow_html=True)
    with cols_k[1]:
        st.markdown(kpi_html("Saldo > R$ 500", formatar_moeda(total_alto),
                             f"{len(df_alto)} clientes", "neg", "negative"),
                    unsafe_allow_html=True)
    with cols_k[2]:
        st.markdown(kpi_html("Saldo Medio", formatar_moeda(saldo_medio),
                             card_class=""), unsafe_allow_html=True)
    with cols_k[3]:
        maior_saldo = df_pos['saldo'].max() if not df_pos.empty else 0
        st.markdown(kpi_html("Maior Saldo", formatar_moeda(maior_saldo),
                             card_class="accent"), unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        section_header("Top Clientes por Saldo em CC",
                       "Clientes com maior recurso parado")

        df_rank_s = df_pos.nlargest(20, 'saldo').copy()

        colors = [CORES['negative'] if s > 500 else CORES['warning']
                  for s in df_rank_s['saldo']]

        fig_s = go.Figure(go.Bar(
            x=safe_numeric(df_rank_s, 'saldo'),
            y=df_rank_s['nome'],
            orientation='h',
            text=df_rank_s['saldo'].apply(formatar_moeda),
            textposition='auto',
            textfont=dict(size=10),
            marker_color=colors,
        ))
        fig_s.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                            xaxis=dict(visible=False))
        apply_plotly_style(fig_s, height=max(400, len(df_rank_s) * 24))
        fig_s.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        # Distribuição por assessor
        if has_col(df_pos, 'assessor'):
            section_header("Saldo CC por Assessor")

            df_s_assessor = df_pos.groupby('assessor').agg(
                saldo=('saldo', 'sum'),
                qtd=('saldo', 'count'),
            ).reset_index().sort_values('saldo', ascending=False)

            fig_sa = px.pie(
                df_s_assessor, values='saldo', names='assessor',
                color_discrete_sequence=COLOR_SEQ,
                hole=0.4,
            )
            fig_sa.update_traces(textposition='inside', textinfo='percent+label',
                                 textfont_size=11)
            apply_plotly_style(fig_sa, height=300)
            st.plotly_chart(fig_sa, use_container_width=True)

        # Distribuição por faixa de saldo
        section_header("Distribuicao por Faixa de Saldo")
        bins = [0, 100, 500, 1000, 5000, 10000, 50000, float('inf')]
        labels = ['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K-50K', '50K+']
        df_pos['faixa_saldo'] = pd.cut(df_pos['saldo'], bins=bins, labels=labels, right=False)
        df_faixa = df_pos.groupby('faixa_saldo', observed=True)['saldo'].count().reset_index()
        df_faixa.columns = ['Faixa', 'Qtd']

        fig_faixa = go.Figure(go.Bar(
            x=df_faixa['Faixa'],
            y=df_faixa['Qtd'],
            text=df_faixa['Qtd'],
            textposition='outside',
            marker_color=CORES['primary'],
        ))
        apply_plotly_style(fig_faixa, height=250)
        fig_faixa.update_layout(
            xaxis=dict(title="Faixa de Saldo (R$)"),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig_faixa, use_container_width=True)

    # --- Tabela completa ---
    section_header("Tabela Completa de Saldos")
    df_tbl_s = df_pos.sort_values('saldo', ascending=False).copy()
    cols_tbl = ['nome', 'saldo']
    rename_tbl = {'nome': 'Cliente', 'saldo': 'Saldo'}
    if has_col(df_tbl_s, 'assessor'):
        cols_tbl.append('assessor')
        rename_tbl['assessor'] = 'Assessor'
    if has_col(df_tbl_s, 'data_referencia'):
        cols_tbl.append('data_referencia')
        rename_tbl['data_referencia'] = 'Data Ref.'

    df_tbl_show = df_tbl_s[cols_tbl].copy()
    df_tbl_show['saldo'] = df_tbl_show['saldo'].apply(formatar_moeda)
    df_tbl_show = df_tbl_show.rename(columns=rename_tbl)
    st.dataframe(df_tbl_show, use_container_width=True, hide_index=True, height=400)


# ================== TAB 6: BASE DE CLIENTES ==================
def render_clientes_premium(df_clientes: pd.DataFrame, filtro_assessor: str):
    """Tab Base de Clientes."""
    if df_clientes.empty:
        st.info("Nenhum dado de clientes encontrado.")
        return

    df = filtrar_por_assessor(df_clientes, filtro_assessor)

    # --- Busca ---
    col_search, col_download = st.columns([3, 1])
    with col_search:
        busca = st.text_input("Buscar cliente:", placeholder="Digite o nome do cliente...")
    with col_download:
        st.markdown("<br>", unsafe_allow_html=True)
        csv = df.to_csv(index=False, sep=';', decimal=',')
        st.download_button("Exportar CSV", csv, "clientes_netuno.csv", "text/csv")

    if busca:
        df = df[df['nome'].str.contains(busca, case=False, na=False)]

    # --- KPIs ---
    cols_k = st.columns(4)
    with cols_k[0]:
        st.markdown(kpi_html("Total Clientes", str(len(df))), unsafe_allow_html=True)
    with cols_k[1]:
        st.markdown(kpi_html("PL Total", formatar_moeda(safe_sum(df, 'pl_total'))),
                    unsafe_allow_html=True)
    with cols_k[2]:
        ticket = safe_sum(df, 'pl_total') / len(df) if len(df) > 0 else 0
        st.markdown(kpi_html("Ticket Medio", formatar_moeda(ticket)),
                    unsafe_allow_html=True)
    with cols_k[3]:
        if has_col(df, 'cidade'):
            st.markdown(kpi_html("Cidades", str(df['cidade'].nunique())),
                        unsafe_allow_html=True)
        elif has_col(df, 'assessor'):
            st.markdown(kpi_html("Assessores", str(df['assessor'].nunique())),
                        unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # --- Distribuição por Faixa ---
    if has_col(df, 'faixa_cliente'):
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            section_header("Distribuicao por Faixa de Cliente")
            df_faixa_c = df.groupby('faixa_cliente').agg(
                qtd=('nome', 'count'),
                pl_total=('pl_total', 'sum'),
            ).reset_index().sort_values('pl_total', ascending=False)

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Bar(
                x=df_faixa_c['faixa_cliente'],
                y=df_faixa_c['qtd'],
                name='Qtd Clientes',
                marker_color=CORES['primary'],
                text=df_faixa_c['qtd'],
                textposition='outside',
            ))
            apply_plotly_style(fig_fc, height=300)
            fig_fc.update_layout(
                xaxis=dict(tickangle=-30),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=10, b=60),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        with col_f2:
            section_header("PL por Faixa de Cliente")
            fig_fc2 = go.Figure(go.Bar(
                x=df_faixa_c['faixa_cliente'],
                y=df_faixa_c['pl_total'],
                text=df_faixa_c['pl_total'].apply(formatar_moeda),
                textposition='outside',
                textfont=dict(size=10),
                marker_color=CORES['secondary'],
            ))
            apply_plotly_style(fig_fc2, height=300)
            fig_fc2.update_layout(
                xaxis=dict(tickangle=-30),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=10, b=60),
            )
            st.plotly_chart(fig_fc2, use_container_width=True)

    # --- Distribuição geográfica ---
    if has_col(df, 'cidade'):
        section_header("Distribuicao Geografica", "Top cidades por qtd de clientes")
        df_cidade = df.groupby('cidade').agg(
            qtd=('nome', 'count'),
            pl_total=('pl_total', 'sum'),
        ).reset_index().sort_values('qtd', ascending=False).head(15)

        fig_cid = go.Figure(go.Bar(
            x=df_cidade['qtd'],
            y=df_cidade['cidade'],
            orientation='h',
            text=df_cidade['qtd'],
            textposition='auto',
            marker_color=CORES['primary'],
        ))
        fig_cid.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                              xaxis=dict(visible=False))
        apply_plotly_style(fig_cid, height=max(250, len(df_cidade) * 24))
        fig_cid.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cid, use_container_width=True)

    # --- Tabela completa ---
    section_header("Base Completa de Clientes")

    # Seleciona e renomeia colunas disponíveis
    col_map = {
        'nome': 'Cliente',
        'assessor': 'Assessor',
        'pl_total': 'PL Total',
        'renda_fixa': 'Renda Fixa',
        'renda_variavel': 'Renda Variavel',
        'fundos': 'Fundos',
        'previdencia': 'Previdencia',
        'cidade': 'Cidade',
        'faixa_cliente': 'Faixa',
        'conta': 'Conta',
    }

    cols_avail = [c for c in col_map.keys() if has_col(df, c)]
    df_display = df[cols_avail].sort_values('pl_total', ascending=False).copy()

    for c in ['pl_total', 'renda_fixa', 'renda_variavel', 'fundos', 'previdencia']:
        if c in df_display.columns:
            df_display[c] = df_display[c].apply(formatar_moeda)

    df_display = df_display.rename(columns=col_map)
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)


# ================== TAB EXTRA: EVOLUÇÃO PL ==================
def render_evolucao_pl(df_pl: pd.DataFrame, df_clientes: pd.DataFrame):
    """Tab Evolução do PL ao longo do tempo."""
    if df_pl.empty:
        st.info("Nenhum dado historico de PL encontrado.")
        return

    if has_col(df_pl, 'data_referencia') and has_col(df_pl, 'pl_total'):
        df_pl['data_referencia'] = pd.to_datetime(df_pl['data_referencia'], errors='coerce')
        df_pl['pl_total'] = safe_numeric(df_pl, 'pl_total')

        # PL agregado por data
        section_header("Evolucao do PL Total", "Patrimonio liquido ao longo do tempo")

        df_evo = df_pl.groupby('data_referencia')['pl_total'].sum().reset_index()
        df_evo = df_evo.sort_values('data_referencia')

        fig_evo = go.Figure()
        fig_evo.add_trace(go.Scatter(
            x=df_evo['data_referencia'],
            y=df_evo['pl_total'],
            mode='lines+markers',
            line=dict(color=CORES['primary'], width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(0, 172, 173, 0.1)',
            name='PL Total',
        ))
        apply_plotly_style(fig_evo, height=380)
        fig_evo.update_layout(
            yaxis=dict(title="PL Total (R$)"),
            xaxis=dict(title="Data"),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig_evo, use_container_width=True)

        # Variação entre datas
        if len(df_evo) >= 2:
            ultimo = df_evo.iloc[-1]['pl_total']
            anterior = df_evo.iloc[-2]['pl_total']
            variacao = ultimo - anterior
            variacao_pct = (variacao / anterior * 100) if anterior > 0 else 0

            cols_ev = st.columns(4)
            with cols_ev[0]:
                st.markdown(kpi_html("PL Atual", formatar_moeda(ultimo)),
                            unsafe_allow_html=True)
            with cols_ev[1]:
                st.markdown(kpi_html("PL Anterior", formatar_moeda(anterior)),
                            unsafe_allow_html=True)
            with cols_ev[2]:
                cls = "pos" if variacao >= 0 else "neg"
                st.markdown(kpi_html("Variacao R$", formatar_moeda(variacao),
                                     delta_class=cls),
                            unsafe_allow_html=True)
            with cols_ev[3]:
                cls = "pos" if variacao_pct >= 0 else "neg"
                st.markdown(kpi_html("Variacao %", formatar_pct(variacao_pct),
                                     delta_class=cls),
                            unsafe_allow_html=True)

        # PL por cliente ao longo do tempo (top 5)
        if has_col(df_pl, 'nome') or has_col(df_pl, 'conta'):
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            section_header("Evolucao PL - Top Clientes")

            nome_col = 'nome' if has_col(df_pl, 'nome') else 'conta'

            # Top 5 clientes pelo PL mais recente
            ultima_data = df_pl['data_referencia'].max()
            df_ultimo = df_pl[df_pl['data_referencia'] == ultima_data]
            top5 = df_ultimo.nlargest(5, 'pl_total')[nome_col].tolist()

            df_top5 = df_pl[df_pl[nome_col].isin(top5)].copy()

            fig_top5 = go.Figure()
            for i, cliente in enumerate(top5):
                df_c = df_top5[df_top5[nome_col] == cliente].sort_values('data_referencia')
                fig_top5.add_trace(go.Scatter(
                    x=df_c['data_referencia'],
                    y=df_c['pl_total'],
                    name=str(cliente)[:30],
                    mode='lines+markers',
                    line=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)], width=2),
                    marker=dict(size=5),
                ))

            apply_plotly_style(fig_top5, height=350)
            fig_top5.update_layout(
                yaxis=dict(title="PL (R$)"),
                xaxis=dict(title="Data"),
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig_top5, use_container_width=True)
    else:
        st.info("Dados historicos de PL nao possuem as colunas necessarias "
                "(data_referencia, pl_total).")


# ================== MAIN ==================
def main():
    """Funcao principal."""
    if not check_password():
        return

    render_header()

    # Carrega dados
    with st.spinner("Carregando dados do Supabase..."):
        df_clientes = carregar_clientes()
        df_saldo = carregar_saldo_cc()
        df_pl = carregar_pl_total()
        df_posicao = carregar_posicao()
        df_proventos = carregar_proventos()

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 15px 0;">
            <h2 style="color: {CORES['primary']}; margin: 0;">🔱 NETUNO</h2>
            <p style="color: {CORES['neutral']}; font-size: 12px; margin: 0;">
                Performance Comercial
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Filtro assessor
        st.markdown(f"**Filtros**")
        assessores = ["Todos"]
        if not df_clientes.empty and has_col(df_clientes, 'assessor'):
            assessores += sorted(df_clientes['assessor'].dropna().unique().tolist())
        filtro_assessor = st.selectbox("Assessor", assessores)

        st.markdown("---")

        # Info
        st.markdown(f"**Dados Carregados**")
        st.caption(f"Clientes: {len(df_clientes)}")
        st.caption(f"Saldos CC: {len(df_saldo)}")
        st.caption(f"Historico PL: {len(df_pl)}")
        st.caption(f"Posicoes: {len(df_posicao)}")
        st.caption(f"Proventos: {len(df_proventos)}")

        st.markdown("---")
        st.caption(f"Atualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

        if st.button("Atualizar Dados", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ---- KPIs Principais ----
    df_clientes_filtrado = filtrar_por_assessor(df_clientes, filtro_assessor)
    render_kpis_principais(df_clientes_filtrado)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ---- Tabs ----
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Performance Comercial",
        "Proventos & Amortizacoes",
        "Posicao & Estrategia",
        "PL & Analise Pareto",
        "Saldo em CC",
        "Base de Clientes",
        "Evolucao PL",
    ])

    with tab1:
        render_performance(df_clientes_filtrado, df_saldo)

    with tab2:
        render_proventos_premium(df_proventos, df_clientes)

    with tab3:
        render_posicao_estrategia(df_posicao, df_clientes, filtro_assessor)

    with tab4:
        render_pareto(df_clientes_filtrado, df_posicao)

    with tab5:
        render_saldo_cc_premium(df_saldo, df_clientes)

    with tab6:
        render_clientes_premium(df_clientes, filtro_assessor)

    with tab7:
        render_evolucao_pl(df_pl, df_clientes)

    # ---- Footer ----
    st.markdown("---")
    st.markdown(
        f"<p style='text-align: center; color: {CORES['neutral']}; font-size: 11px;'>"
        f"Netuno Dados Premium | Grupo Netuno • R2F Capital • {datetime.now().year}"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
