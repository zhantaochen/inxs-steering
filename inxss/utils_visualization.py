import matplotlib.pyplot as plt
import numpy as np

def arc_arrow(ax, start_angle=0., end_angle=np.pi/2, radius=1., arrow_size=0.05, color='k', lw=1.5, unit='degree', label=None):

    if unit == 'degree':
        start_angle = start_angle / 180. * np.pi
        end_angle = end_angle / 180. * np.pi
    base_angle = end_angle - arrow_size * np.sqrt(3) / radius
    
    # This draws the arc
    theta = np.linspace(start_angle, base_angle, 100)
    radii = radius * np.ones_like(theta)
    ax.plot(theta, radii, lw=lw, color=color, zorder=30, clip_on=False)
    
    # Calculate vertices for the arrow tip
    vertices = [(end_angle, radius), (base_angle, radius + arrow_size), (base_angle, radius - arrow_size)]

    arrow = plt.Polygon(vertices, closed=True, color=color)
    ax.add_patch(arrow)
    arrow.set_clip_on(False)
    arrow.set_zorder(30)
    
    # if label is not None:
    #     ax.text(start_angle - 5/180 * np.pi, radius, label, ha='left', va='bottom', fontsize=12)
    if label is not None:
        label_radius = radius - 2 * arrow_size
        label_angle = end_angle
        ax.text(label_angle, label_radius, label, 
                ha='left', va='top', 
                rotation=np.degrees(label_angle) - 90,
                rotation_mode='anchor',
                fontsize=12)

def rad_arrow(ax, angle=90., start_radius=1., end_radius=1.2, arrow_size=0.05, color='k', lw=1.5, unit='degree', label=None):

    if unit == 'degree':
        angle = angle / 180. * np.pi
    base_radius = end_radius - arrow_size * np.sqrt(3)
    
    # This draws the arc
    radii = np.linspace(start_radius, base_radius, endpoint=True)
    theta = angle * np.ones_like(radii)
    ax.plot(theta, radii, lw=lw, color=color, zorder=30, clip_on=False)
    

    # Calculate vertices for the arrow tip
    vertices = [(angle - arrow_size/base_radius, base_radius), (angle, base_radius + arrow_size), (angle+arrow_size/base_radius, base_radius)]

    arrow = plt.Polygon(vertices, closed=True, color=color)
    ax.add_patch(arrow)
    arrow.set_clip_on(False)
    arrow.set_zorder(30)
    
    # if label is not None:
    #     ax.text(start_angle - 5/180 * np.pi, radius, label, ha='left', va='bottom', fontsize=12)
    if label is not None:
        label_radius = end_radius
        label_angle = angle - 2 * arrow_size / label_radius
        ax.text(label_angle, label_radius+0.05, label, 
                ha='left', va='top', 
                rotation=np.degrees(angle) - 90,
                rotation_mode='anchor',
                fontsize=12)
        
def visualize_utility(angle, utility, ax=None, plot_max=True):
    
    utility = (utility - utility.min()) / (utility.max() - utility.min())
    
    if ax is None:
        # Create a figure and an axis with a polar projection
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
    # Plot data on the polar axis
    ax.plot(angle / 360 * 2 * np.pi, utility, color='C0', linewidth=2)
    arc_arrow(ax, start_angle=55, end_angle=80, radius=utility.max() * 1, arrow_size=0.025, label='$\psi$')
    rad_arrow(ax, angle=135/2, start_radius=utility.max() * 0.85, end_radius=utility.max() * 1.15, arrow_size=0.025, label='$U(\psi)$')

    if plot_max:
        ax.plot([angle[utility.argmax()] / 360 * 2 * np.pi,]*2, [0, utility.max()], color='C1', linestyle='--')
    
    ax.set_rlim([0, 1.35])
    ax.set_rticks([0, 0.5, 1])

    return ax


def update_projection(ax, axi, projection='3d', fig=None):
    """https://stackoverflow.com/a/75485793 
    """
    if fig is None:
        fig = plt.gcf()
    rows, cols, start, stop = axi.get_subplotspec().get_geometry()
    ax.flat[start].remove()
    ax.flat[start] = fig.add_subplot(rows, cols, start+1, projection=projection)