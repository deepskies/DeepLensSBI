import matplotlib.pyplot as plt

def plot_residuals(observed_image, model_image):
    fig, ax = plt.subplots(1,3,figsize=(15,5))

    #This is the observed image
    ax[0].set_title("Observed")
    ax[0].imshow(observed_image)

    #This is the model image
    ax[1].set_title("Best Fit")
    ax[1].imshow(model_image)

    #This is the residuals
    ax[2].set_title("Residuals")
    resid = ax[2].imshow(model_image-observed_image)
    fig.colorbar(resid, ax=ax, shrink=0.7)

    return fig, ax