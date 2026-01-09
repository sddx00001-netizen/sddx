import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Set page configuration
st.set_page_config(page_title="Gradient & Steepest Ascent Visualizer", layout="wide")
st.title("Calculus Visualization: Gradient & Steepest Ascent")

st.markdown("""
This application helps visualize **functions of two variables**, their **partial derivatives**, 
and the **gradient vector** to understand the direction of **steepest ascent**.
""")

# --- Sidebar: User Input ---
st.sidebar.header("1. Input Function & Point")
st.sidebar.info("Tips: Use 'x' and 'y'. E.g., `x**2 + y**2` or `x * exp(-(x**2 + y**2))`")

# 1. User inputs the function
function_input = st.sidebar.text_input("Enter a function f(x, y):", value="x**2 + y**2")

# 2. User inputs the point (x0, y0)
x_val = st.sidebar.number_input("x coordinate:", value=1.0, step=0.1)
y_val = st.sidebar.number_input("y coordinate:", value=1.0, step=0.1)

# --- Core Calculation (using SymPy) ---
try:
    x, y = sp.symbols('x y')
    f_expr = sp.sympify(function_input)
    
    # Calculate Partial Derivatives
    fx = sp.diff(f_expr, x)
    fy = sp.diff(f_expr, y)
    
    # Create lambda functions for numerical evaluation
    f_func = sp.lambdify((x, y), f_expr, 'numpy')
    fx_func = sp.lambdify((x, y), fx, 'numpy')
    fy_func = sp.lambdify((x, y), fy, 'numpy')
    
    # Evaluate at the specific point
    z0 = float(f_func(x_val, y_val))
    fx0 = float(fx_func(x_val, y_val))
    fy0 = float(fy_func(x_val, y_val))
    
    # Gradient Vector Calculation
    gradient_vector = [fx0, fy0]
    magnitude = np.sqrt(fx0**2 + fy0**2)

    # --- Display Section ---
    
    # Part 1: Mathematical Analysis
    st.header("Part 1: Mathematical Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Partial Derivatives")
        st.latex(r"f(x, y) = " + sp.latex(f_expr))
        st.markdown("**Partial w.r.t x:**")
        st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(fx))
        st.markdown("**Partial w.r.t y:**")
        st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy))
        st.caption("Partial derivatives represent the rate of change along the x and y axes.")

    with col2:
        st.subheader("2. Gradient Calculation")
        st.markdown(f"At point $({x_val}, {y_val})$:")
        st.latex(r"\nabla f = \left< \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right>")
        st.latex(rf"\nabla f({x_val}, {y_val}) = \left< {fx0:.2f}, {fy0:.2f} \right>")
        st.metric("Gradient Magnitude", f"{magnitude:.4f}")

    with col3:
        st.subheader("3. Steepest Ascent")
        st.info("""
        **Key Concept:**
        The gradient vector $\\nabla f$ always points in the direction of **steepest ascent** (maximum increase) of the function.
        """)
        st.write(f"Direction: **<{fx0:.2f}, {fy0:.2f}>**")

    # Part 2: Visualization
    st.header("Part 2: Visualization")
    
    # Prepare data for plotting
    range_val = 2.5
    X = np.linspace(x_val - range_val, x_val + range_val, 50)
    Y = np.linspace(y_val - range_val, y_val + range_val, 50)
    X, Y = np.meshgrid(X, Y)
    Z = f_func(X, Y)
    
    vis_col1, vis_col2 = st.columns(2)
    
    # Plot 1: 2D Contour Plot with Gradient Arrow
    with vis_col1:
        st.subheader("2D Contour Plot & Gradient")
        fig1, ax1 = plt.subplots()
        cp = ax1.contourf(X, Y, Z, cmap='viridis', levels=20)
        plt.colorbar(cp)
        # Plot the point
        ax1.plot(x_val, y_val, 'ro') 
        # Plot the gradient arrow
        ax1.arrow(x_val, y_val, fx0*0.5, fy0*0.5, head_width=0.2, head_length=0.2, fc='r', ec='r')
        ax1.set_title("Gradient Arrow (Red) Points Uphill")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig1)
        st.caption("The red arrow shows the gradient direction starting from the red dot. Notice it crosses contour lines perpendicularly.")

    # Plot 2: 3D Surface Plot
    with vis_col2:
        st.subheader("3D Surface Plot")
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
        # Plot point in 3D
        ax2.scatter(x_val, y_val, z0, color='r', s=50, label='Point')
        # Plot gradient direction (quiver)
        ax2.quiver(x_val, y_val, z0, fx0, fy0, 0, length=1, color='r', linewidth=2, label='Gradient Direction')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax2.set_title(f"Surface of f(x,y)")
        st.pyplot(fig2)
        st.caption("The 3D view of the mountain. The gradient tells you which way is 'up'.")

except Exception as e:
    st.error(f"Error parsing function. Please check syntax. Error: {e}")

# --- Footer: Helper for assignment ---
st.divider()
st.markdown("### How to use this for your report:")
st.markdown("""
1.  **Screenshot 1**: Use the default example ($x^2 + y^2$). It's a simple 'bowl'. Explain how the gradient points outward/upward.
2.  **Screenshot 2**: Enter a complex function like `x * exp(-(x**2 + y**2))` (Peak/Valley). Explain how the gradient changes direction depending on where you are on the hill.
""")
