import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Add cells to the notebook
cells.append(nbf.v4.new_markdown_cell("# Logarithms - Python Implementation\nConverted from MATLAB script by Jack Baker (2/2/2024)"))

cells.append(nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams"""))

cells.append(nbf.v4.new_markdown_cell("## Define color specifications"))

cells.append(nbf.v4.new_code_cell("""colorspec = [
    np.array([56, 95, 150])/255,  # First color (blue)
    np.array([207, 89, 33])/255   # Second color (orange)
]"""))

cells.append(nbf.v4.new_markdown_cell("## Create x values and compute exponentials"))

cells.append(nbf.v4.new_code_cell("""x = np.arange(-1, 1.1, 0.1)
eX = np.exp(x)
tenX = 10**x

def format_figure():
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()"""))

cells.append(nbf.v4.new_markdown_cell("## Exponentials on semilogy plot"))

cells.append(nbf.v4.new_code_cell("""plt.figure()
plt.semilogy(x, eX, color=colorspec[0], linewidth=2, label='$e^x$')
plt.semilogy(x, tenX, color=colorspec[1], linewidth=2, label='$10^x$')
yl = plt.ylim()
plt.plot([0, 0], yl, '-k')  # Vertical line at x=0
plt.xlabel('x')
plt.legend(loc='northwest')
format_figure()
plt.title('Exponential Functions (Log Scale)')
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## Exponentials on linear plot"))

cells.append(nbf.v4.new_code_cell("""plt.figure()
plt.plot(x, eX, color=colorspec[0], linewidth=2, label='$e^x$')
plt.plot(x, tenX, color=colorspec[1], linewidth=2, label='$10^x$')
yl = plt.ylim()
plt.plot([0, 0], yl, '-k')  # Vertical line at x=0
plt.xlabel('x')
plt.legend(loc='northwest')
format_figure()
plt.title('Exponential Functions (Linear Scale)')
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## Create x values for logarithms and compute logs"))

cells.append(nbf.v4.new_code_cell("""x_log = np.arange(0.1, 10.1, 0.1)  # Starting from 0.1 to avoid log(0)
lnX = np.log(x_log)
log10X = np.log10(x_log)"""))

cells.append(nbf.v4.new_markdown_cell("## Logarithms on linear plot"))

cells.append(nbf.v4.new_code_cell("""plt.figure()
plt.plot(x_log, lnX, color=colorspec[0], linewidth=2, label='$\ln(x)$')
plt.plot(x_log, log10X, color=colorspec[1], linewidth=2, label='$\log_{10}(x)$')
plt.plot(x_log, np.zeros_like(x_log), '-k')  # Horizontal line at y=0
plt.xlabel('x')
plt.legend(loc='northwest')
format_figure()
plt.title('Logarithm Functions (Linear Scale)')
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## Logarithms on semilogx plot"))

cells.append(nbf.v4.new_code_cell("""plt.figure()
plt.semilogx(x_log, lnX, color=colorspec[0], linewidth=2, label='$\ln(x)$')
plt.semilogx(x_log, log10X, color=colorspec[1], linewidth=2, label='$\log_{10}(x)$')
plt.semilogx(x_log, np.zeros_like(x_log), '-k')  # Horizontal line at y=0
plt.xlabel('x')
plt.legend(loc='northwest')
format_figure()
plt.title('Logarithm Functions (Log Scale)')
plt.show()"""))

nb['cells'] = cells

# Write the notebook to a file
with open('logarithms.ipynb', 'w') as f:
    nbf.write(nb, f)