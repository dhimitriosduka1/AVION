import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame="circle"):
    """Create a radar chart with `num_vars` axes."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


# Data
categories = [
    "QFVS\n(avg F-score)",
    "EK-100\nMIR\n(mAP)",
    "EK-100\nMIR\n(nDCG)",
    "EgoTaskQA\n(acc.)",
    "EgoMQ\n(R@5\nIoU@0.3)",
    "CharadesEgo\n(mAP)",
    "EgoNLQ\n(R@5\nIoU@0.3)",
    "EgoMCQ\n(intra-vid.\nacc.)",
]

# Values for each model
EgoVLPv2 = [52.1, 47.3, 61.9, 37.9, 68.2, 34.1, 23.8, 60.9]
EgoVLPv1 = [49.7, 45.0, 59.4, 32.7, 65.6, 32.1, 18.8, 57.2]

# Number of variables
N = len(categories)
theta = radar_factory(N, frame="polygon")

# Create figure
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="radar"))
fig.subplots_adjust(top=0.95, bottom=0.05)

# Plot data
ax.plot(theta, EgoVLPv2, "o-", linewidth=2, color="#7FCBC4", label="EgoVLPv2")
ax.fill(theta, EgoVLPv2, alpha=0.25, color="#7FCBC4")

ax.plot(theta, EgoVLPv1, "o-", linewidth=2, color="#B8B8B8", label="EgoVLPv1 [57]")
ax.fill(theta, EgoVLPv1, alpha=0.25, color="#B8B8B8")

# Set labels
ax.set_varlabels(categories)

# Set y-axis limits and ticks
ax.set_ylim(0, 75)
ax.set_yticks([18.8, 32.7, 45.0, 57.2, 68.2])
ax.set_yticklabels(["18.8", "32.7", "45.0", "57.2", "68.2"])

# Add grid
ax.grid(True, linestyle="-", linewidth=0.5, color="gray", alpha=0.3)

# Customize tick labels - increase pad to move category labels further from contour
ax.tick_params(axis="x", which="major", labelsize=11, pad=20)
ax.tick_params(axis="y", which="major", labelsize=11)
for label in ax.get_xticklabels():
    label.set_fontsize(12)

# Add legend
legend = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=12)

# # Add values on the plot
# for i, (angle, v2, v1) in enumerate(zip(theta, EgoVLPv2, EgoVLPv1)):
#     # Position for EgoVLPv2 values (outer)
#     ax.text(
#         angle,
#         v2 + 2,
#         f"{v2}",
#         ha="center",
#         va="center",
#         fontsize=10,
#         fontweight="bold",
#         color="#2A7F7A",
#     )
#     # Position for EgoVLPv1 values (inner)
#     if abs(v2 - v1) > 3:  # Only show if values are different enough
#         ax.text(
#             angle,
#             v1 - 2,
#             f"{v1}",
#             ha="center",
#             va="center",
#             fontsize=9,
#             color="#666666",
#         )

plt.tight_layout()
plt.savefig("radar_chart.png", dpi=300, bbox_inches="tight")
print("Radar chart saved successfully!")
