#!/usr/bin/env python3
"""
Machine learning for PDF shapes
"""
# ========================================================================
#
# Imports
#
# ========================================================================
import os
import glob
import time
import datetime
import numpy as np
from numpy import linalg as la
import pickle
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

# ========================================================================
def create_logdir(model_name):
    """Create a log directory for a model"""
    time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.abspath(os.path.join("runs", f"{time}_{model_name}"))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir

# ========================================================================
def SplitInputOutput(Train, Validate, coVariance=True):
    """
    Generate scaled training, dev, test arrays
    """

    if(coVariance):
        x_vars = ['MeanA', 'VarA', 'MeanB', 'VarB', 'CovAB']
    else:
        x_vars = ['MeanA', 'VarA', 'MeanB', 'VarB']
    
    y_vars = [col for col in Train if col.startswith("Y")]

    Xtrain = Train.loc[:, x_vars]
    Ytrain = Train.loc[:, y_vars]

    Xvalidate = Validate.loc[:, x_vars]
    Yvalidate = Validate.loc[:, y_vars]

    # Scale the data
    scaler = RobustScaler()
    scaler.fit(Xtrain)
    
    Xtrain = pd.DataFrame(scaler.transform(Xtrain),
                          index=Xtrain.index,
                          columns=Xtrain.columns)
    Xvalidate = pd.DataFrame(scaler.transform(Xvalidate),
                             index=Xvalidate.index,
                             columns=Xvalidate.columns)
    if(coVariance):
        joblib.dump(scaler, "DNN_robustScalerMultiScalarPDF.pkl")
    else:
        joblib.dump(scaler, "DNN_robustScalerMultiScalarPDF_NocoVar_May2_NoDelta.pkl")

    return Xtrain, Ytrain, Xvalidate, Yvalidate, scaler


# ========================================================================
def jensen_shannon_divergence(p, q):
    """
    This will be part of scipy as some point.
    See https://github.com/scipy/scipy/pull/8295
    We use this implementation for now:
    https://stackoverflow.com/questions/15880133/jensen-shannon-divergence

    :param p: PDF (normalized to 1)
    :type p: array
    :param q: PDF (normalized to 1)
    :type q: array
    """
    eps = 1e-13
    M = np.clip(0.5 * (p + q), eps, None)
    return 0.5 * (stats.entropy(p, M) + stats.entropy(q, M))


# ========================================================================
def calculate_jsd(y, yp):
    """
    Calculate the JSD metric on each PDF prediction
    """
    y = np.asarray(y, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    return np.array(
        [jensen_shannon_divergence(y[i, :], yp[i, :]) for i in
         range(y.shape[0])]
    )


# ========================================================================
def summarize_training(ytrain, mtrain, ydev, mdev, fname="summary.log"):
    """
    Summarize training

    :param label: method label
    :type label: string
    :param ytrain: true training values
    :type ytrain: array
    :param mtrain: predicted training values
    :type mtrain: array
    :param ydev: true dev values
    :type ydev: array
    :param mdev: predicted dev values
    :type mdev: array
    :param fname: log filename
    :type fname: str
    """

    jsd_train = calculate_jsd(ytrain, mtrain)
    jsd_dev = calculate_jsd(ydev, mdev)
    std_error_train = np.std(np.ravel(ytrain - mtrain) ** 2)
    std_error_dev = np.std(np.ravel(ydev - mdev) ** 2)

    percentiles = [85, 90, 95]
    percentiles_train = np.percentile(jsd_train, percentiles)
    percentiles_dev = np.percentile(jsd_dev, percentiles)

    msg = (
        f"""Training data errors\n"""
        f"""  MAE: {mean_absolute_error(ytrain, mtrain):e}\n"""
        f"""  MSE: {mean_squared_error(ytrain, mtrain):e}\n"""
        f"""  std SE: {std_error_train:e}\n"""
        f"""  R^2: {r2_score(ytrain, mtrain):.2f}\n"""
        f"""  JSD 85 percentile: {percentiles_train[0]:5f}\n"""
        f"""  JSD 90 percentile: {percentiles_train[1]:5f}\n"""
        f"""  JSD 95 percentile: {percentiles_train[2]:5f}\n"""
        f"""\n"""
        f"""Dev data errors\n"""
        f"""  MAE: {mean_absolute_error(ydev, mdev):e}\n"""
        f"""  MSE: {mean_squared_error(ydev, mdev):e}\n"""
        f"""  std SE: {std_error_dev:e}\n"""
        f"""  R^2: {r2_score(ydev, mdev):.2f}\n"""
        f"""  JSD 85 percentile: {percentiles_dev[0]:5f}\n"""
        f"""  JSD 90 percentile: {percentiles_dev[1]:5f}\n"""
        f"""  JSD 95 percentile: {percentiles_dev[2]:5f}\n"""
    )

    # Output and write to file
    print(msg)
    with open(fname, "w") as f:
        f.write(msg)


# =====================     
def plotAllMeans(path):
    plt.close()
    fig = plt.figure()
    Pkfiles = glob.glob('./run_*/PDF_w64_s64/*.gz', recursive=True)
    for file in Pkfiles:
        PDF = pd.read_pickle(file, compression='infer')
        plt.plot(PDF["MeanA"], PDF["MeanB"], "ko",markersize=0.25)
    PDF = pd.read_pickle("SyntheticDeltaFunctions.gz", compression='infer')
    plt.plot(PDF["MeanA"], PDF["MeanB"], "ro",markersize=0.25)
    plt.show()
    

# ===========================================================
def generate_training_samples(path, nTrain, nValidate):

    Train = pd.DataFrame()
    Validate = pd.DataFrame()
    AllPDFs = pd.DataFrame()
    
    Pkfiles = glob.glob('./run_*/PDF_w64_s64/*.gz', recursive=True)
    nValidate = nTrain + nValidate
    for file in Pkfiles:
        print("Reading file {0}".format(file))
        PDF = pd.read_pickle(file, compression='infer')
        # split the rows in 3 ways with nTrain and nValidate rows
        T, V, R = np.split(PDF.sample(frac=1), [nTrain, nValidate], axis=0)
        Train = Train.append(T)
        Validate = Validate.append(V)
    PDF = pd.read_pickle("SyntheticDeltaFunctions.gz", compression='infer')
    print("Reading file {0}".format(PDF.shape))
    Train = Train.append(PDF)

    Train = Train.sample(frac=1)
    Validate = Validate.sample(frac=1)

    return Train, Validate

# ===========================================================
def generate_training_samples_allpdf(path, nTrain, nValidate):

    Train = pd.DataFrame()
    Validate = pd.DataFrame()
    AllPDFs = pd.DataFrame()
    Pkfiles = glob.glob('./run_*/PDF_w64_s64/*.gz', recursive=True)
    nValidate = nTrain + nValidate
    for file in Pkfiles:
        print("Reading file {0}".format(file))
        PDF = pd.read_pickle(file, compression='infer')
        AllPDFs = AllPDFs.append(PDF.sample(frac=1))
    PDF = pd.read_pickle("SyntheticDeltaFunctions.gz", compression='infer')
    AllPDFs = AllPDFs.append(PDF.sample(frac=1))
    T, V, R = np.split(AllPDFs.sample(frac=1), [nTrain, nValidate], axis=0)
    Train = Train.append(T)
    Validate = Validate.append(V)

    Train = Train.sample(frac=1)
    Validate = Validate.sample(frac=1)

    return Train, Validate

# ==========================================================
def L1epsilonError(YDNS, YRF, Aij):

    return np.sum(np.absolute(np.subtract(YDNS, YRF)))

# ==========================================================
def plotTimeVarPDF_DNN(path, scalerfile, DNN, nSample=8,
                       nAbins=64, nBbins=64, coVariance=True):
    x = np.linspace(0, 1.0, nAbins)
    y = np.linspace(0, 1.0, nBbins)
    xv, yv = np.meshgrid(x, y)

    Aij = (x[2]-x[1])*(y[2]-y[1])

    scaler = joblib.load(scalerfile)

    Pkfiles = os.path.join(path, "*.gz")
    for file in glob.glob(Pkfiles):
        pdf_pages = PdfPages(file.replace('gz','pdf'))
        Pkl = pd.read_pickle(file, compression='infer')
        y_vars = [col for col in Pkl if col.startswith("Y")]
        DNS = Pkl.loc[:,y_vars]
        if(coVariance):
            x_vars = ['MeanA', 'VarA', 'MeanB', 'VarB', 'CovAB']
        else:
            x_vars = ['MeanA', 'VarA', 'MeanB', 'VarB']
        
        inputDNS = Pkl.loc[:, x_vars]
        
        inputDNS = pd.DataFrame(scaler.transform(inputDNS),
                          index=inputDNS.index,
                          columns=inputDNS.columns)

        print("Predicting PDFs from {0}".format(file))
        Predict = DNN.predict(inputDNS)
        
        plt.close('all')

        L1Errnorm = np.empty(DNS.shape[0])
        JSDErr = np.empty(DNS.shape[0])
        
        for itime in range(DNS.shape[0]):

            L1Errnorm[itime] = L1epsilonError(np.array(DNS.iloc[itime,:]),
                                              Predict[itime,:], Aij)
            JSDErr[itime] = jensen_shannon_divergence(np.array(DNS.iloc[itime,:]),
                                                      Predict[itime,:])
            
            DNS_PDF = np.reshape(np.array(DNS.iloc[itime,:]),
                                 (nAbins, nBbins), order='F')

            RF_PDF = np.reshape(Predict[itime,:],
                                (nAbins, nBbins), order='F')
            
            if(itime%nSample == 0):
                fig, (ax1, ax2) = plt.subplots(1, 2)
                
                ax1.contourf(xv, yv, DNS_PDF, cmap='RdBu_r')
                pointsO = [[1.0,0], [0,1.0], [1.0,1.0]]
                pointsI = [[1.0,0], [0,1.0], [0,0]]
                tri2 = plt.Polygon(pointsO, fc='white', ec = 'white',closed=None)
                tri1 = plt.Polygon(pointsI, ec = 'black',fill=None)
                ax1.spines['right'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                ax1.add_patch(tri2)
                ax1.add_patch(tri1)
                ax1.set_title('DNS')

                ax2.contourf(xv, yv, RF_PDF, cmap='RdBu_r')
                pointsO = [[1.0,0], [0,1.0], [1.0,1.0]]
                pointsI = [[1.0,0], [0,1.0], [0,0]]
                tri2 = plt.Polygon(pointsO, fc='white', ec = 'white',closed=None)
                tri1 = plt.Polygon(pointsI, ec = 'black',fill=None)
                ax2.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax2.add_patch(tri2)
                ax2.add_patch(tri1)
                ax2.set_title('DNN Predict')
                
                pdf_pages.savefig(fig)

        time = np.arange(DNS.shape[0])
        fig1, (ax3, ax4) = plt.subplots(1, 2)

        ax3.plot(time, L1Errnorm, 'k-')
        ax3.set_xlabel('time step')
        ax3.set_ylabel('L1 norm of error between DNS and DNN')

        ax4.plot(time, JSDErr, 'r-')
        ax4.set_xlabel('time step')
        ax4.set_ylabel('JSD between DNS and DNN')

        plt.tight_layout()
        
        pdf_pages.savefig(fig1)

        pdf_pages.close()

# ========================================================================
# Fully connected NN
class Net(nn.Module):
    def __init__(self, input_size, layer_sizes, vh=None):
        super(Net, self).__init__()

        if vh is None:
            self.vh = VariableHandler()
        else:
            self.vh = vh

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(
            zip([input_size] + layer_sizes[:-1], layer_sizes)
        ):
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(
                    name="L%i" % (i), module=nn.Linear(in_size, out_size)
                )
                self.MLP.add_module(name="A%i" % (i), module=nn.LeakyReLU())
                self.MLP.add_module(name="B%i" % (i), module=nn.BatchNorm1d(out_size))
            else:
                self.MLP.add_module(
                    name="L%i" % (i), module=nn.Linear(in_size, out_size)
                )
                self.MLP.add_module(name="softmax", module=nn.Softmax(dim=1))

    def forward(self, x):
        return self.MLP(x)

    def predict(self, X, batch_size=64):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        meval = np.zeros((n, self.layer_sizes[-1]))
        for batch, i in enumerate(range(0, n, batch_size)):
            slc = np.s_[i : i + batch_size, :]
            meval[slc] = self.forward(self.vh.tovar(X[slc])).cpu().data.numpy()
        return meval

    def load(self, fname):
        """Load pickle file containing model"""
        self.load_state_dict(
            torch.load(fname, map_location=lambda storage, loc: storage)
        )
        self.eval()


# ========================================================================
# Torch Variable handler
class VariableHandler:
    def __init__(self, device=torch.device("cpu"), dtype=torch.float):
        self.device = device
        self.dtype = dtype

    def tovar(self, input):
        return Variable(torch.as_tensor(input, dtype=self.dtype, device=self.device))


# ========================================================================
def dnn_training(Xtrain, Xdev, Ytrain, Ydev, use_gpu=False):
    """
    Train using a deep neural network
    """

    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.double
    vh = VariableHandler(device=device, dtype=dtype)

    # Make sure inputs are numpy arrays 
    Xtrain = np.asarray(Xtrain, dtype=np.float64)
    Ytrain = np.asarray(Ytrain, dtype=np.float64)
    Xdev = np.asarray(Xdev, dtype=np.float64)
    Ydev = np.asarray(Ydev, dtype=np.float64)

    # N is batch size; D_in is input dimension; D_out is output dimension
    batch_size = 128
    input_size = Xtrain.shape[1]
    layer_sizes = [256, 512, 2048, Ytrain.shape[1]]
    torch.manual_seed(5465462)

    # Construct the NN model
    model = Net(input_size, layer_sizes, vh).to(device=device, dtype=dtype)

    # The number of times entire dataset is trained
    nepochs = 1250

    # Learning rate
    learning_rate = 1e-4

    # Loss and optimizer
    criterion = nn.BCELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Tensorboard output
    writer = SummaryWriter()
    logdir = writer.file_writer.get_logdir()
    model_name = "DNN_noDelta"
    xdummy = vh.tovar(torch.randn(1, Xtrain.shape[1]))
    writer.add_graph(model, (xdummy,), verbose=True)

    # Train the model
    nbatches = Xtrain.shape[0] // batch_size
    for epoch in range(nepochs):

        model.train()
        permutation = torch.randperm(Xtrain.shape[0])

        for batch, i in enumerate(range(0, Xtrain.shape[0], batch_size)):

            # Global step
            step = epoch * nbatches + batch

            # Take a batch
            indices = permutation[i : i + batch_size]
            batch_x = vh.tovar(Xtrain[indices, :])
            batch_y = vh.tovar(Ytrain[indices, :])

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(batch_x)

            # Compute and log information
            loss = criterion(y_pred, batch_y)
            writer.add_scalar("loss", loss.item(), step)
            if batch % 10 == 0:

                print(
                    "Epoch [{0:d}/{1:d}], Batch [{2:d}/{3:d}], Loss: {4:.4e}".format(
                        epoch + 1, nepochs, batch + 1, nbatches, loss.item()
                    )
                )

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loss and adaptive time step
        model.eval()
        val_loss = criterion(model(vh.tovar(Xdev)), vh.tovar(Ydev))
        writer.add_scalar("val_loss", val_loss.item(), step)
        print(
            "Epoch [{0:d}/{1:d}], Validation loss: {2:.4e}".format(
                epoch + 1, nepochs, val_loss.item()
            )
        )
        for param_group in optimizer.param_groups:
            print("Current learning rate", param_group["lr"])

        # Save the models
        torch.save(model.state_dict(), os.path.join(logdir, model_name + ".pkl"))

    writer.close()

    model.eval()
    mtrain = model.predict(Xtrain)
    mdev = model.predict(Xdev)

    # Summarize training
    summarize_training(
        Ytrain, mtrain, Ydev, mdev, os.path.join(logdir, model_name + ".log")
    )

    return mtrain, mdev, model


# ==========================================================
def plotTimeVarML(path, DNN4scalerFile, DNN5scalerFile, RF4scalerFile, RF5scalerFile,
                  DNN4file, RF4file, DNN5file, RF5file,
                  nAbins=64, nBbins=64, coVariance=True):

    x = np.linspace(0, 1.0, nAbins)
    y = np.linspace(0, 1.0, nBbins)
    xv, yv = np.meshgrid(x, y)

    DNN4scaler = joblib.load(DNN4scalerFile)
    DNN5scaler = joblib.load(DNN5scalerFile)
    RF4scaler = joblib.load(RF4scalerFile)
    RF5scaler = joblib.load(RF5scalerFile)

    RF_covar = joblib.load(RF5file)
    RF_NoCovar = joblib.load(RF4file)

    device = torch.device("cpu")
    dtype = torch.double
    vh = VariableHandler(device=device, dtype=dtype)

    # Construct the NN model
    input_size = 4
    layer_sizes = [256, 512, 2048, 4096]
    model_NoCovar = Net(input_size, layer_sizes, vh).to(device=device, dtype=dtype)
    DNN_NoCovar = model_NoCovar.load(DNN4file)
    
    # Construct the NN model
    input_size = 5
    vh = VariableHandler(device=device, dtype=dtype)
    model_covar = Net(input_size, layer_sizes, vh).to(device=device, dtype=dtype)
    DNN_covar = model_covar.load(DNN5file)

    plotTimes = [0.2, 0.4, 0.7, 1.0, 1.5, 2.0]
    tauEddy = 0.2
    dt = 0.01

    TimeStep = np.int32(np.divide(np.multiply(plotTimes,tauEddy), dt))
    
    Pkfiles = os.path.join(path, "*.gz")
    for file in glob.glob(Pkfiles):
        filePlot = file.replace('gz','pdf')
        filePlot = filePlot.replace('PaperData', 'PaperPlots/Jan27')
        pdf_pages = PdfPages(filePlot)
        
        Pkl = pd.read_pickle(file, compression='infer')
        y_vars = [col for col in Pkl if col.startswith("Y")]
        DNS = Pkl.loc[:,y_vars]

        # First with CoVariance
        x_vars = ['MeanA', 'VarA', 'MeanB', 'VarB', 'CovAB']
        inputDNS = Pkl.loc[:, x_vars]
        inputDNS = pd.DataFrame(scaler.transform(inputDNS),
                          index=inputDNS.index,
                          columns=inputDNS.columns)

        print("Predicting PDFs from {0}".format(file))
        # DNN
        DNN_predict_covar = DNN_covar.predict(inputDNS)
        # RF
        RF_predict_covar = RF_covar.predict(inputDNS)

        # Second without CoVariance
        x_vars = ['MeanA', 'VarA', 'MeanB', 'VarB']
        inputDNS = Pkl.loc[:, x_vars]
        inputDNS = pd.DataFrame(scaler.transform(inputDNS),
                          index=inputDNS.index,
                          columns=inputDNS.columns)

        print("Predicting PDFs from {0}".format(file))
        # DNN
        DNN_predict_NoCovar = DNN_NoCovar.predict(inputDNS)
        # RF
        RF_predict_NoCovar = RF_NoCovar.predict(inputDNS)
        
        for itime in TimeStep:

            DNS_PDF = np.reshape(np.array(DNS.iloc[itime,:]),
                                 (nAbins, nBbins), order='F')

            RF_NoCovar_PDF = np.reshape(RF_predict_NoCovar[itime,:],(nAbins, nBbins), order='F')
            DNN_NoCovar_PDF = np.reshape(DNN_predict_NoCovar[itime,:],(nAbins, nBbins), order='F')

            RF_Covar_PDF = np.reshape(RF_predict_covar[itime,:],(nAbins, nBbins), order='F')
            DNN_Covar_PDF = np.reshape(DNN_predict_covar[itime,:],(nAbins, nBbins), order='F')

            plt.close('all')
            
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)    
            plt.contourf(xv, yv, DNS_PDF, cmap='RdBu_r')
            pointsO = [[1.0,0], [0,1.0], [1.0,1.0]]
            pointsI = [[1.0,0], [0,1.0], [0,0]]
            tri2 = plt.Polygon(pointsO, fc='white', ec = 'white',closed=None)
            tri1 = plt.Polygon(pointsI, ec = 'black',fill=None)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.add_patch(tri2)
            ax.add_patch(tri1)
            ax.set_title('DNS')
            
            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)    
            plt.contourf(xv, yv, RF_NoCovar_PDF, cmap='RdBu_r')
            pointsO = [[1.0,0], [0,1.0], [1.0,1.0]]
            pointsI = [[1.0,0], [0,1.0], [0,0]]
            tri2 = plt.Polygon(pointsO, fc='white', ec = 'white',closed=None)
            tri1 = plt.Polygon(pointsI, ec = 'black',fill=None)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.add_patch(tri2)
            ax.add_patch(tri1)
            ax.set_title('RF-4')
            pdf_pages.savefig(fig)

            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)    
            plt.contourf(xv, yv, DNN_NoCovar_PDF, cmap='RdBu_r')
            pointsO = [[1.0,0], [0,1.0], [1.0,1.0]]
            pointsI = [[1.0,0], [0,1.0], [0,0]]
            tri2 = plt.Polygon(pointsO, fc='white', ec = 'white',closed=None)
            tri1 = plt.Polygon(pointsI, ec = 'black',fill=None)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.add_patch(tri2)
            ax.add_patch(tri1)
            ax.set_title('DNN-4')
            pdf_pages.savefig(fig)

            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)    
            plt.contourf(xv, yv, RF_Covar_PDF, cmap='RdBu_r')
            pointsO = [[1.0,0], [0,1.0], [1.0,1.0]]
            pointsI = [[1.0,0], [0,1.0], [0,0]]
            tri2 = plt.Polygon(pointsO, fc='white', ec = 'white',closed=None)
            tri1 = plt.Polygon(pointsI, ec = 'black',fill=None)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.add_patch(tri2)
            ax.add_patch(tri1)
            ax.set_title('RF-5')
            pdf_pages.savefig(fig)

            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)    
            plt.contourf(xv, yv, DNN_Covar_PDF, cmap='RdBu_r')
            pointsO = [[1.0,0], [0,1.0], [1.0,1.0]]
            pointsI = [[1.0,0], [0,1.0], [0,0]]
            tri2 = plt.Polygon(pointsO, fc='white', ec = 'white',closed=None)
            tri1 = plt.Polygon(pointsI, ec = 'black',fill=None)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.add_patch(tri2)
            ax.add_patch(tri1)
            ax.set_title('DNN-5')
            pdf_pages.savefig(fig)


# ==========================================================
def DumpJITscipt(DNN4scalerFile, DNN4file, OutFile, coVariance=False):
    device = torch.device("cpu")
    dtype = torch.double
    vh = VariableHandler(device=device, dtype=dtype)
    input_size = 4
    layer_sizes = [256, 512, 2048, 4096]
    model_NoCovar = Net(input_size, layer_sizes, vh).to(device=device, dtype=dtype)
    DNN_NoCovar = model_NoCovar.load(DNN4file)
    scaler = joblib.load(DNN4scalerFile)
    xinput = [3.33357768e-01, 2.09439868e-04, 3.32701194e-01, 1.39771614e-04]
    xinput = np.reshape(xinput, (1,4))
    Xscale = scaler.transform(xinput)
    traced_script_module = torch.jit.trace(model_NoCovar,torch.as_tensor(Xscale))
    traced_script_module.save(OutFile)
