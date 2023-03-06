import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set_style("ticks")

from sklearn.neighbors import KernelDensity

s = pd.read_csv("substructures2.txt", delimiter="|", header=7)

ra_s = s["RA"]
dec_s = s["DEC"]
z_s = s["Redshift"]
obj_name = s["Object Name"]

table = pd.read_csv("tables_photometric/Hydra-Centaurus-probgal&isoarea.csv",
                    usecols=["RA", "DEC", "PROB_GAL", "zml", "r_petro", "g_petro"])

ra = table["RA"]
dec = table["DEC"]
probgal = table["PROB_GAL"]
zml = table["zml"]
rpetro = table["r_petro"]
gpetro = table["g_petro"]

mlim = 20
ra_max = 180
ra_min = 150
dec_min = -48
dec_max = -20

mask = (rpetro < mlim) & (zml > 0.005) & (zml < 0.03) & (ra < ra_max) & (ra
                    > ra_min) & (dec > dec_min) & (dec < dec_max)

mask_s = (ra_s < ra_max) & (ra_s > ra_min) & (z_s < 0.02) & (dec_s > dec_min) & (dec_s < dec_max)

x = np.deg2rad(ra[mask]) #like longitude
y = np.deg2rad(dec[mask]) #like latitude

xbin_size = 200
ybin_size = 200

xbins=xbin_size*1j
ybins=ybin_size*1j

xx, yy = np.mgrid[x.min():x.max():xbins, y.min():y.max():ybins]

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, kernel='epanechnikov', metric='haversine',
                                                    algorithm="ball_tree", **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

bandwidth = 0.015
xx, yy, zz = kde2D(x, y, bandwidth=bandwidth, xbins=xbins, ybins=ybins)

cl_names = ["N3393", "A1060", "N3054", "N3087", "Antlia", "N3347", "N3250", "N3256", "N3263"]

cl_ras = [162.09, 159.17, 148.61, 149.78, 157.51, 160.69, 156.63, 156.96, 157.30]

cl_decs = [-25.16, -27.52, -25.70, -34.22, -35.32, -36.35, -39.94, -43.90, -44.12]

fontsize = 15
labelsize = 15

fig = plt.figure(figsize=(20, 40))
ax = fig.add_subplot(111)
sc1 = ax.pcolormesh(np.rad2deg(xx), np.rad2deg(yy), zz, cmap='gist_stern')
ax.scatter(np.rad2deg(x), np.rad2deg(y), s=0.5, color='white')
ax.set_title("Bandwidth = {:.4f}".format(bandwidth), fontsize=fontsize)
ax.set_ylabel("DEC (deg)", fontsize=fontsize)
ax.set_xlabel("RA (deg)", fontsize=fontsize)
ax.set_title("r_lim = {:.2f}".format(mlim), fontsize=fontsize)

ax.scatter(x=ra_s[mask_s], y=dec_s[mask_s], s=50, color="black", marker="d")
ax.contour(np.rad2deg(xx), np.rad2deg(yy), zz, colors='white', levels=10, extend='min', width=0.5)


from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(sc1, cax=cax1, orientation='vertical')
cbar.set_label(label="Projected density", fontsize=fontsize)
cbar.ax.tick_params(labelsize=labelsize)

# ax.legend(fontsize=fontsize*0.8)
ax.xaxis.set_tick_params(labelsize=labelsize, width=5)
ax.yaxis.set_tick_params(labelsize=labelsize, width=5)

dx = 0.5
# for ra, dec, text in zip(ra_s[m], dec_s[m], obj_name[m]):
#     ax.text(ra-dx, dec-dx, s=text, fontsize=30, fontweight='bold', color='yellow')

for ra_i, dec_i, text in zip(cl_ras, cl_decs, cl_names):
    ax.scatter(ra_i, dec_i, marker='o', s = 100, color="yellow")
    ax.text(ra_i-dx, dec_i-dx, s=text, fontsize=10, fontweight='bold', color='yellow')

ax.invert_xaxis()
plt.show()
