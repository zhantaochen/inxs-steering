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
        ax.text(label_angle, label_radius, label, 
                ha='left', va='top', 
                rotation=np.degrees(angle) - 90,
                rotation_mode='anchor',
                fontsize=12)