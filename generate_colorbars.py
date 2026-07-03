import os
import matplotlib.pyplot as plt
import matplotlib as mpl

def draw_colorbar(vmin, vmax, ticks, cmap_name, filename):
    # Set Arial font properties
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.unicode_minus': False,
    })
    
    # 100mm width = 3.937 inches. Slender height = 8mm = 0.315 inches.
    width_inch = 140 / 25.4
    height_inch = 2 / 25.4
    
    # Create figure and axes specifically for the colorbar
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    
    # Set normalization and colormap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    
    # Draw colorbar base
    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal',
        ticks=ticks
    )
    
    # Set tick labels font size
    cb.ax.tick_params(labelsize=8)
    
    # Save the figure
    os.makedirs("legends", exist_ok=True)
    out_path = os.path.join("legends", filename)
    fig.savefig(out_path, dpi=1200, transparent=True, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f"[*] Saved colorbar to {out_path}")

def main():
    # 1. Flow field value colorbar (0 to 5, RdBu_r)
    draw_colorbar(
        vmin=0, 
        vmax=5, 
        ticks=[0, 1, 2, 3, 4, 5], 
        cmap_name="RdBu_r", 
        filename="colorbar_flow_field.png"
    )
    
    # 2. Error value colorbar (-5 to 5, RdBu_r)
    draw_colorbar(
        vmin=-5, 
        vmax=5, 
        ticks=[-5, -2.5, 0, 2.5, 5], 
        cmap_name="RdBu_r", 
        filename="colorbar_error.png"
    )

if __name__ == '__main__':
    main()
