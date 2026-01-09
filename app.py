import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from sympy.abc import x, y

# --- Page Configuration ---
st.set_page_config(page_title="Partial Derivative Explorer", layout="wide")

st.title("Interactive Partial Derivative Explorer")
st.markdown("""
**Assignment Topic:** Partial derivatives as rates of change.
This application visualizes functions of two variables $f(x, y)$ and calculates their partial derivatives dynamically.
It demonstrates how the slope changes in the x and y directions at a specific point.
""")

# --- Sidebar: Inputs and Controls ---
st.sidebar.header("1. Define Function")

# Quick buttons for Assignment Requirement: "2 examples (two different complexities)"
st.sidebar.markdown("**Select Example Complexity:**")
col_a, col_b = st.sidebar.columns(2)
if col_a.button("Simple Example"):
    st.session_state.func_input = "x**2 + y**2"
if col_b.button("Complex Example"):
    st.session_state.func_input = "x * exp(-(x**2 + y**2))"

# Initialize session state if empty
if 'func_input' not in st.session_state:
    st.session_state.func_input = "x**2 + y**2"

# Text input for custom function
func_str = st.sidebar.text_input(
    "Enter a function expression (Python syntax):", 
    value=st.session_state.func_input,
    help="Examples: x**2 + y**2, sin(x)*y, x*y - x**2"
)

st.sidebar.header("2. Set Point Parameters")
x_val = st.sidebar.slider("Value of x", -2.0, 2.0, 1.0, 0.1)
y_val = st.sidebar.slider("Value of y", -2.0, 2.0, 1.0, 0.1)

# --- Core Calculation Logic (SymPy) ---
try:
    # 1. Parse the input function string
    f_expr = sp.sympify(func_str)
    
    # 2. Calculate symbolic partial derivatives
    fx_expr = sp.diff(f_expr, x)  # Partial derivative with respect to x
    fy_expr = sp.diff(f_expr, y)  # Partial derivative with respect to y
    
    # 3. Convert symbolic expressions to numerical Python functions
    f_func = sp.lambdify((x, y), f_expr, 'numpy')
    fx_func = sp.lambdify((x, y), fx_expr, 'numpy')
    fy_func = sp.lambdify((x, y), fy_expr, 'numpy')

    # 4. Calculate values at the specific point (x_val, y_val)
    z_val = f_func(x_val, y_val)
    dz_dx = fx_func(x_val, y_val)
    dz_dy = fy_func(x_val, y_val)

    # --- Display Mathematical Results ---
    st.subheader("1. Mathematical Analysis")
    st.markdown("Here we calculate the exact values of the partial derivatives.")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("##### Function Value")
        st.latex(f"f(x, y) = {sp.latex(f_expr)}")
        st.info(f"f({x_val}, {y_val}) = **{z_val:.4f}**")
        
    with c2:
        st.markdown("##### Partial Derivative (w.r.t x)")
        st.caption("Rate of change in the x-direction")
        st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(fx_expr))
        st.success(f"Result at point: **{dz_dx:.4f}**")

    with c3:
        st.markdown("##### Partial Derivative (w.r.t y)")
        st.caption("Rate of change in the y-direction")
        st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy_expr))
        st.success(f"Result at point: **{dz_dy:.4f}**")

    # --- 3D Visualization (Plotly) ---
    st.subheader("2. 3D Surface Visualization")
    
    # Generate grid data for the plot
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate Z values for the grid safely
    try:
        Z = f_func(X, Y)
    except Exception as e:
        st.error(f"Error calculating grid for visualization: {e}")
        Z = np.zeros_like(X)

    # Create the 3D Surface Plot
    fig = go.Figure(data=[
        go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='Surface')
    ])

    # Add the specific point P (Red Marker)
    fig.add_trace(go.Scatter3d(
        x=[x_val], y=[y_val], z=[z_val],
        mode='markers',
        marker=dict(size=8, color='red', symbol='circle'),
        name='Current Point P'
    ))

    # Update plot layout
    fig.update_layout(
        title=f"Surface Plot of {func_str}",
        autosize=True,
        width=900, height=600,
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis (Function Value)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig)
    
    st.caption("**Instructions:** Click and drag the graph to rotate. Scroll to zoom in/out.")

except Exception as e:
    st.error(f"Input Error: {e}")
    st.warning("Please enter a valid mathematical expression using Python syntax (e.g., x**2 + y**2).")
