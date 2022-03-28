import nibabel as nib
import numpy as np
from preprocess.normalization import normalize_mri
from scipy import spatial
import collections
import h5py
import pickle

class imagepatches(object):

    def __init__(self, fname, mask, label,
                 gm=150, wm=250, csf=10, num_classes=4, pad = 3, masklabel=False,
                 fname_t1=None, k_t2=4, k_t2_init=None,
                 k_t1=2, k_t1_init=None, threedim=False, channels=None, coords=None,
                 origcoords = None, normalize=True, normfactors = None,
                 contlabel = None, spherecoords= None, setbounds = False, bounds=(),
                 interval = 5, numslicex = None, numslicey = None, numslicez =None,
                 fnoutput=None, dataNum = 0, pkl = True, singleslice = False, edgemap = None):

        self.pad = pad
        if not coords is None:
            self.coords = nib.load(coords).get_fdata()
        else:
            self.coords = coords
        self.origcoords = origcoords

        self.masklabel = masklabel
        self.norm = normalize
        self.contlabelfn = contlabel

        self.fname = fname
        self.fname_t1 = fname_t1
        self.fname_mask = mask
        self.spherecoords = spherecoords

        self.dataNum = dataNum
        self.singleslice = singleslice
        self.fnoutput = fnoutput
        self.pkl = pkl
        self.edgemap = edgemap
        if edgemap:
            self.edgemap = nib.load(edgemap).get_fdata()
            self.edgemap[self.edgemap > 0] = 1

        if normfactors:
            self.normfactors = normfactors
        else:
            self.normfactors = None

        self.k_t2 = k_t2
        self.k_t1 = k_t1
        self.k_t2_init=k_t2_init
        # if k_t2_init is None:
        #     self.k_t2_init = [500,100,1000,1500]
        self.k_t1_init = k_t1_init
        if k_t1_init is None:
            self.k_t1_init = [500,250]

        self.nii = nib.load(fname)
        if self.fname_t1:
            self.nii_t1 = nib.load(fname_t1)
            self.data_t1 = self.nii_t1.get_fdata()
        self.data = self.nii.get_fdata()
        if self.spherecoords:
            self.spherecoordsdata = nib.load(self.spherecoords).get_fdata()
        self.dataOrigShape = self.data.shape

        self.indices = []
        self.indices_upsampled = []

        self.nii3 = nib.load(mask)
        self.mask = self.nii3.get_fdata()

        self.gm = gm
        self.wm = wm
        self.csf = csf
        self.label = nib.load(label).get_fdata()
        self.interval = interval

        self.numslicex = numslicex
        self.numslicey = numslicey
        self.numslicez = numslicez

        if self.contlabelfn:
            self.contlabel = nib.load(self.contlabelfn).get_fdata()

        self.label[self.label == 255] = 1
        self.label[self.label == 0] = 4
        self.label[self.label == wm] = 0
        self.label[self.label == gm] = 1
        self.label[self.label == csf] = 2
        self.num_classes = num_classes

        self.indices_upsampled =[]

        self.upsample()
        if setbounds is False:
            self.get_bounds()
        else:
            (self.xpos, self.xpos_end) = bounds[0]
            (self.ypos, self.ypos_end) = bounds[1]
            (self.zpos, self.zpos_end) = bounds[2]

        if threedim:
            self.normalize()
            self.render_patches_3d()
            self.create_data_struct_3d()
        elif not self.origcoords is None:
            self.normalize()
            self.render_patches_simcoords()
            self.create_data_struct()
        elif self.numslicez or self.numslicex or self.numslicey:
            if channels:
                self.channels = channels
            else:
                self.channels = 1
            if self.norm:
                self.normalize()
            self.render_patches_slice()
            self.create_data_struct()
        elif self.fnoutput:
            if self.norm:
                self.normalize()
            self.render_data_struct()
        elif channels:
            self.channels = channels
            self.render_patches_channels()
            self.create_data_struct()
        else:
            if self.norm:
                self.normalize()
            self.render_patches()
            self.create_data_struct()

    def upsample(self):
        # self.data = scipy.ndimage.zoom(self.data, (self.upsamplefactor,self.upsamplefactor,self.upsamplefactor),
        #                                order=self.order)
        #
        # self.mask = scipy.ndimage.zoom(self.mask, (self.upsamplefactor, self.upsamplefactor, self.upsamplefactor),
        #                                        order=0)
        self.mask[self.mask >0] =1
        self.dataUpsampledShape = self.data.shape

    def normalize(self):
        if self.normfactors:
            self.mean, self.std = self.normfactors
        else:
            self.mean, self.std = normalize_mri(self.data, self.mask, self.k_t2, self.k_t2_init)
        idxs = np.where(self.data > 0)
        self.data = self.data.astype(np.float64)
        self.data[idxs] -= self.mean
        self.data[idxs] /= self.std

        if self.fname_t1:
            self.mean_t1, self.std_t1 = normalize_mri(self.data_t1, self.mask, self.k_t1, self.k_t1_init)
            self.data_t1 = self.data_t1.astype(np.float64)
            self.data_t1[idxs] -= self.mean_t1
            self.data_t1[idxs] /= self.std_t1

    ## get bounds
    def get_bounds(self):
        for i in range(self.mask.shape[0]):
            if np.sum(self.mask[i, :,:]) != 0:
                self.xpos = i
                break

        for i in range(self.mask.shape[0]-1, 0, -1):
            if np.sum(self.mask[i, :,:]) != 0:
                self.xpos_end = i+1
                break

        for i in range(self.mask.shape[1]):
            if np.sum(self.mask[:, i,:]) != 0:
                self.ypos = i
                break

        for i in range(self.mask.shape[1]-1, 0, -1):
            if np.sum(self.mask[:, i,:]) != 0:
                self.ypos_end = i+1
                break

        for i in range(self.mask.shape[2]):
            if np.sum(self.mask[:, :,i]) != 0:
                self.zpos = i
                break

        for i in range(self.mask.shape[2]-1, 0, -1):
            if np.sum(self.mask[:, :,i]) != 0:
                self.zpos_end = i+1
                break

    def render_patches_slice(self):
        pad = self.pad
        N = int(np.count_nonzero(self.mask))
        self.X = np.zeros([N,1])
        self.X5 = np.zeros([N])
        if self.spherecoords:
            self.sphericalcoordinates = np.zeros([N,3])
        self.neighbors = np.zeros([N, 2*pad+1, 2*pad+1, self.channels])
        self.neighbors_z = np.zeros([N, 2 * pad + 1, 2 * pad + 1, self.channels])
        self.neighbors_y = np.zeros([N, 2 * pad + 1, 2 * pad + 1, self.channels])

        n = 0

        ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]

        zs = self.zpos_end - self.zpos
        if self.numslicez is None:
            zinterval = 1
        else:
            zinterval = int(np.round(zs/self.numslicez))
        ys = self.ypos_end - self.ypos
        if self.numslicey is None:
            yinterval = 1
        else:
            yinterval = int(np.round(ys / self.numslicey))
        xs = self.xpos_end - self.xpos
        if self.numslicex is None:
            xinterval = 1
        else:
            xinterval = int(np.round(xs / self.numslicex))

        if self.channels == 1:
            self.data = np.expand_dims(self.data, axis=3)

        for k in np.arange(self.zpos, self.zpos_end, zinterval):
            for i in range(self.xpos, self.xpos_end):
                for j in range(self.ypos, self.ypos_end):
                    # print(i,j,k)
                    padslice = self.data[i-pad:i+pad+1, j-pad:j+pad+1, k]
                    padzslice = self.data[i-pad:i+pad+1, j, k-pad:k+pad+1]
                    padyslice = self.data[i , j- pad:j + pad + 1, k - pad:k + pad + 1]

                    boolean = True
                    if self.masklabel:
                        boolean = (self.label[i,j,k] != 4)
                    if (self.mask[i,j,k] > 0) & boolean:

                        self.indices.append(ravel((i,j,k), self.data.shape))
                        self.X5[n] = self.label[i, j, k]
                        if self.spherecoords:
                            self.sphericalcoordinates[n] = self.spherecoordsdata[i,j,k]

                        self.neighbors[n, :, :] = padslice
                        self.neighbors_z[n, :, :] = padzslice
                        self.neighbors_y[n, :, :] = padyslice

                        n+=1

        for j in np.arange(self.ypos, self.ypos_end, yinterval):
            for i in range(self.xpos, self.xpos_end):
                for k in range(self.zpos, self.zpos_end):
                    padslice = self.data[i - pad:i + pad + 1, j - pad:j + pad + 1, k]
                    padzslice = self.data[i - pad:i + pad + 1, j, k - pad:k + pad + 1]
                    padyslice = self.data[i, j - pad:j + pad + 1, k - pad:k + pad + 1]

                    boolean = True
                    if self.masklabel:
                        boolean = (self.label[i, j, k] != 4)
                    if (self.mask[i, j, k] > 0) & boolean:

                        self.indices.append(ravel((i, j, k), self.data.shape))
                        self.X5[n] = self.label[i, j, k]
                        if self.spherecoords:
                            self.sphericalcoordinates[n] = self.spherecoordsdata[i, j, k]

                        self.neighbors[n, :, :] = padslice
                        self.neighbors_z[n, :, :] = padzslice
                        self.neighbors_y[n, :, :] = padyslice

                        n += 1

        for i in np.arange(self.xpos, self.xpos_end, xinterval):
            for k in range(self.zpos, self.zpos_end):
                for j in range(self.ypos, self.ypos_end):
                    padslice = self.data[i - pad:i + pad + 1, j - pad:j + pad + 1, k]
                    padzslice = self.data[i - pad:i + pad + 1, j, k - pad:k + pad + 1]
                    padyslice = self.data[i, j - pad:j + pad + 1, k - pad:k + pad + 1]

                    boolean = True
                    if self.masklabel:
                        boolean = (self.label[i, j, k] != 4)
                    if (self.mask[i, j, k] > 0) & boolean:

                        self.indices.append(ravel((i, j, k), self.data.shape))
                        self.X5[n] = self.label[i, j, k]
                        if self.spherecoords:
                            self.sphericalcoordinates[n] = self.spherecoordsdata[i, j, k]

                        self.neighbors[n, :, :] = padslice
                        self.neighbors_z[n, :, :] = padzslice
                        self.neighbors_y[n, :, :] = padyslice

                        n += 1

        self.indices = np.asarray(self.indices)
        print('Image patches generated.')

    def render_patches(self):
        pad = self.pad
        ## break down into patches
        # approximate array allocation
        N = int(np.count_nonzero(self.mask))
        self.X = np.zeros([N,1])
        self.X5 = np.zeros([N])
        if self.spherecoords:
            self.sphericalcoordinates = np.zeros([N,3])
        if self.contlabelfn:
            self.CL = np.zeros([N])
        self.neighbors = np.zeros([N, 2*pad+1, 2*pad+1])
        self.neighbors_z = np.zeros([N, 2 * pad + 1, 2 * pad + 1])
        self.neighbors_y = np.zeros([N, 2 * pad + 1, 2 * pad + 1])
        # self.circle = np.zeros([N, self.r * 4])
        if not self.coords is None:
            self.coordsvec = np.zeros(N)

        if self.fname_t1:
            self.neighbors_t1 =np.zeros([N, 2*pad+1, 2*pad+1])
            self.neighbors_z_t1 = np.zeros([N, 2 * pad + 1, 2 * pad + 1])
            self.neighbors_y_t1 = np.zeros([N, 2 * pad + 1, 2 * pad + 1])

        n = 0

        ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]

        for k in np.arange(self.zpos, self.zpos_end, self.interval):
            testslice = self.data[:,:,k]
            maskslice = self.mask[:,:,k]
            labelslice = self.label[:,:,k]

            if self.fname_t1:
                t1slice = self.data_t1[:,:,k]

            i = self.xpos
            while i < self.xpos_end:
                j = self.ypos
                while j < self.ypos_end:
                    padslice = testslice[i-pad:i+pad+1, j-pad:j+pad+1]
                    padzslice = self.data[i-pad:i+pad+1, j, k-pad:k+pad+1]
                    padyslice = self.data[i , j- pad:j + pad + 1, k - pad:k + pad + 1]

                    boolean = ( 1== 1)
                    if self.masklabel:
                        boolean = (labelslice[i,j] != 4)
                    if (maskslice[i,j] > 0) & boolean:

                        # ULupsampled = ravel((i,j,k), self.data.shape)

                        self.indices.append(ravel((i,j,k), self.data.shape))
                        self.X5[n] = self.label[i, j, k]
                        if self.spherecoords:
                            self.sphericalcoordinates[n] = self.spherecoordsdata[i,j,k]

                        if self.contlabelfn:
                            self.CL[n] = self.contlabel[i,j,k]
                        if not self.coords is None:
                            self.coordsvec[n] = self.coords[i, j, k]

                        self.neighbors[n, :, :] = padslice
                        self.neighbors_z[n, :, :] = padzslice
                        self.neighbors_y[n, :, :] = padyslice

                        if self.fname_t1:
                            self.neighbors_t1[n, :, :] = t1slice[i-pad:i+pad+1, j-pad:j+pad+1]
                            self.neighbors_z_t1[n, :, :] = self.data_t1[i-pad:i+pad+1, j, k-pad:k+pad+1]
                            self.neighbors_y_t1[n, :, :] = self.data_t1[i , j- pad:j + pad + 1, k - pad:k + pad + 1]

                        n += 1
                    j += self.interval
                i += self.interval

        self.indices = np.asarray(self.indices)
        print('Image patches generated.')

    def render_patches_simcoords(self):
        pad = self.pad
        N = int(np.count_nonzero(self.mask))
        self.X5 = np.zeros([N])
        self.neighbors = np.zeros([N, 2*pad+1, 2*pad+1])
        self.neighbors_z = np.zeros([N, 2 * pad + 1, 2 * pad + 1])
        self.neighbors_y = np.zeros([N, 2 * pad + 1, 2 * pad + 1])
        # self.circle = np.zeros([N, self.r * 4])
        if not self.coords is None:
            self.coordsvec = np.zeros(N)

        if self.fname_t1:
            self.neighbors_t1 =np.zeros([N, 2*pad+1, 2*pad+1])
            self.neighbors_z_t1 = np.zeros([N, 2 * pad + 1, 2 * pad + 1])
            self.neighbors_y_t1 = np.zeros([N, 2 * pad + 1, 2 * pad + 1])

        n = 0

        ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]
        tmp = self.coords.ravel().reshape(-1, 1)
        tree = spatial.KDTree(tmp)
        for coordidx, origcoord in enumerate(self.origcoords):

            _, tmpidx = tree.query(np.array([origcoord]))

            # i, j, k = np.where(self.coords == origcoord)
            i,j,k = np.unravel_index(tmpidx, self.dataOrigShape)

            if self.fname_t1:
                t1slice = self.data_t1[:,:,k]

            padslice = self.data[i - pad:i + pad + 1, j - pad:j + pad + 1, k]
            padzslice = self.data[i - pad:i + pad + 1, j, k - pad:k + pad + 1]
            padyslice = self.data[i, j - pad:j + pad + 1, k - pad:k + pad + 1]

            self.indices.append(ravel((i, j, k), self.data.shape))

            self.X5[n] = self.label[i, j, k]
            if not self.coords is None:
                self.coordsvec[n] = self.coords[i, j, k]

            self.neighbors[n, :, :] = padslice
            self.neighbors_z[n, :, :] = padzslice
            self.neighbors_y[n, :, :] = padyslice

            if self.fname_t1:
                self.neighbors_t1[n, :, :] = t1slice[i - pad:i + pad + 1, j - pad:j + pad + 1]
                self.neighbors_z_t1[n, :, :] = self.data_t1[i - pad:i + pad + 1, j, k - pad:k + pad + 1]
                self.neighbors_y_t1[n, :, :] = self.data_t1[i, j - pad:j + pad + 1, k - pad:k + pad + 1]

            n +=1

        self.indices = np.asarray(self.indices)
        print('Image patches generated.')

    def render_patches_3d(self):
        pad = self.pad
        ## break down into patches
        # approximate array allocation
        N = int(np.count_nonzero(self.mask))
        self.X5 = np.zeros([N])
        self.neighbors = np.zeros([N, 2*pad+1, 2*pad+1, 2*pad+1])

        if self.fname_t1:
            self.neighbors_t1 =np.zeros([N, 2*pad+1, 2*pad+1, 2*pad+1])

        n = 0

        ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]

        for k in range(self.zpos, self.zpos_end):
            maskslice = self.mask[:,:,k]
            labelslice = self.label[:,:,k]

            i = self.xpos
            while i < self.xpos_end:
                j = self.ypos
                while j < self.ypos_end:
                    padsvol = self.data[i-pad:i+pad+1, j-pad:j+pad+1, k-pad:k+pad+1]

                    boolean = ( 1== 1)
                    if self.masklabel:
                        boolean = (labelslice[i,j] != 4)
                    if (maskslice[i,j] > 0) & boolean:

                        # ULupsampled = ravel((i,j,k), self.data.shape)

                        self.indices.append(ravel((i,j,k), self.data.shape))
                        self.X5[n] = self.label[i, j, k]
                        self.neighbors[n] = padsvol

                        if self.fname_t1:
                            self.neighbors_t1[n] = self.data_t1[i-pad:i+pad+1, j-pad:j+pad+1,k-pad:k+pad+1]

                        n += 1
                    j += 1
                i +=1

        self.indices = np.asarray(self.indices)
        print('3D Image patches generated.')

    def render_patches_channels(self):
        pad = self.pad
        ## break down into patches
        # approximate array allocation
        N = int(np.count_nonzero(self.mask))
        self.X5 = np.zeros([N])
        if self.contlabelfn:
            self.CL = np.zeros([N])
        if self.spherecoords:
            self.sphericalcoordinates = np.zeros([N, 3])
        self.neighbors = np.zeros([N, 2*pad+1, 2*pad+1, self.channels])
        self.neighbors_z = np.zeros([N, 2 * pad + 1, 2 * pad + 1, self.channels])
        self.neighbors_y = np.zeros([N, 2 * pad + 1, 2 * pad + 1, self.channels])

        if self.fname_t1:
            self.neighbors_t1 =np.zeros([N, 2*pad+1, 2*pad+1, self.channels])
            self.neighbors_z_t1 = np.zeros([N, 2 * pad + 1, 2 * pad + 1, self.channels])
            self.neighbors_y_t1 = np.zeros([N, 2 * pad + 1, 2 * pad + 1, self.channels])

        n = 0

        ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]

        for k in range(self.zpos, self.zpos_end):
            testslice = self.data[:,:,k,:]
            maskslice = self.mask[:,:,k]
            labelslice = self.label[:,:,k]

            if self.fname_t1:
                t1slice = self.data_t1[:,:,k,:]

            i = self.xpos
            while i < self.xpos_end:
                j = self.ypos
                while j < self.ypos_end:
                    padslice = testslice[i-pad:i+pad+1, j-pad:j+pad+1, :]
                    padzslice = self.data[i-pad:i+pad+1, j, k-pad:k+pad+1, :]
                    padyslice = self.data[i , j- pad:j + pad + 1, k - pad:k + pad + 1, :]

                    boolean = ( 1== 1)
                    if self.masklabel:
                        boolean = (labelslice[i,j] != 4)
                    if (maskslice[i,j] > 0) & boolean:

                        # ULupsampled = ravel((i,j,k), self.data.shape)

                        self.indices.append(ravel((i,j,k), self.data.shape))
                        self.X5[n] = self.label[i, j, k]

                        if self.spherecoords:
                            self.sphericalcoordinates[n] = self.spherecoordsdata[i,j,k]

                        if self.contlabelfn:
                            self.CL[n] = self.contlabel[i,j,k]

                        self.neighbors[n, :, :, :] = padslice
                        self.neighbors_z[n, :, :, :] = padzslice
                        self.neighbors_y[n, :, :, :] = padyslice

                        if self.fname_t1:
                            self.neighbors_t1[n, :, :, :] = t1slice[i-pad:i+pad+1, j-pad:j+pad+1, :]
                            self.neighbors_z_t1[n, :, :, :] = self.data_t1[i-pad:i+pad+1, j, k-pad:k+pad+1, :]
                            self.neighbors_y_t1[n, :, :, :] = self.data_t1[i , j- pad:j + pad + 1, k - pad:k + pad + 1, :]

                        n += 1
                    j += 1
                i +=1

        self.indices = np.asarray(self.indices)
        print('Image patches generated.')

    def create_data_struct(self):
        ### test
        # nonzeros = np.count_nonzero(self.X)
        nonzeros = len(self.indices)

        self.X5 = self.X5[0:nonzeros]
        if self.contlabelfn:
            self.CL = self.CL[0:nonzeros]
        self.neighbors = self.neighbors[0:nonzeros]
        self.neighbors_z = self.neighbors_z[0:nonzeros]
        self.neighbors_y = self.neighbors_y[0:nonzeros]
        if self.spherecoords:
            self.sphericalcoordinates = self.sphericalcoordinates[0:nonzeros]
        # self.line = self.line[0:nonzeros,:]
        # self.zline = self.zline[0:nonzeros, :]
        if not self.coords is None:
            self.coordsvec = self.coordsvec[0:nonzeros]

        sortidxs = np.argsort(self.indices)
        b = self.indices[sortidxs]
        y = [item for item, count in collections.Counter(self.indices).items() if count > 1]
        if len(y) > 0:
            idxs = np.where(np.isin(b, y))
            c = idxs[0][idxs[0] % 2 == 1]

            self.indices = np.delete(b,c)
            self.neighbors = np.delete(self.neighbors[sortidxs],c, axis=0)
            self.neighbors_z = np.delete(self.neighbors_z[sortidxs], c, axis=0)
            self.neighbors_y = np.delete(self.neighbors_y[sortidxs], c, axis=0)
            self.X5 = np.delete(self.X5[sortidxs], c)
            self.X = np.delete(self.X[sortidxs], c)
        #
        if self.fname_t1:
            self.neighbors_t1 = self.neighbors_t1[0:nonzeros]
            self.neighbors_z_t1 = self.neighbors_z_t1[0:nonzeros]
            self.neighbors_y_t1 = self.neighbors_y_t1[0:nonzeros]
        #

        del self.nii3
        del self.data
        del self.mask
        if self.fname_t1:
            del self.data_t1

    def create_data_struct_3d(self):
        ### test
        # nonzeros = np.count_nonzero(self.X)
        nonzeros = len(self.indices)
        self.X5 = self.X5[0:nonzeros]
        self.neighbors = self.neighbors[0:nonzeros, :, :,:]

        if self.fname_t1:
            self.X_t1 = self.X_t1[0:nonzeros]
            self.neighbors_t1 = self.neighbors_t1[0:nonzeros, :, :,:]

        # self.y = np_utils.to_categorical(self.X5, self.num_classes)

        del self.nii
        del self.nii3
        del self.data
        del self.mask
        if self.fname_t1:
            del self.data_t1

    def render_data_struct(self):
        ### test
        # nonzeros = np.count_nonzero(self.X)

        xpos = self.xpos - self.pad
        xposend = self.xpos_end + self.pad
        ypos = self.ypos - (self.pad )
        yposend = self.ypos_end + self.pad
        zpos = self.zpos - (self.pad )
        zposend = self.zpos_end + self.pad

        self.roimask = self.mask[xpos:xposend, ypos:yposend, zpos:zposend]
        tmpmask = self.mask[xpos:xposend, ypos:yposend, zpos:zposend]  # segment out the mask with the boundaries
        tmplabel = self.label[xpos:xposend, ypos:yposend, zpos:zposend]  # segment out label

        if self.masklabel:
            self.indices = np.where((self.mask.ravel() > 0) & (self.label.ravel() != 4))[0]
            self.roicropped_indices = np.where((self.roimask.ravel() > 0) & (tmplabel.ravel() != 4))[0]
        else:
            self.indices = np.where(self.mask.ravel() >0)[0]
            self.roicropped_indices = np.where(self.roimask.ravel() > 0)[0]

        self.mask = np.zeros_like(self.mask)
        self.mask[xpos:xposend, ypos:yposend, zpos:zposend] = 1


        origmask = self.mask[self.xpos:self.xpos_end, self.ypos:self.ypos_end, self.zpos:self.zpos_end]
        origlabel = self.mask[self.xpos:self.xpos_end, self.ypos:self.ypos_end, self.zpos:self.zpos_end]

        if self.masklabel:
            '''get the indices relative to the bounded area only in the areas where it is masked and labeled'''
            self.cropped_indices = np.where((tmpmask.ravel() > 0) & (tmplabel.ravel() != 4))[0]
            ''' get indices in the non-padded areas'''
            self.cropped_indices_nopad = np.where((origmask.ravel() > 0) & (origlabel.ravel() != 4))[0]
        else:
            self.cropped_indices  = np.where(tmpmask.ravel() >0)[0]
            self.cropped_indices_nopad = np.where((origmask.ravel() > 0))[0]

        '''get the mapping according to the 1D version of the cropped indices so that the
        we can select the indices to move to in the training dataset (the cropped indices are
        relative to the cropped region, whereas the this will return a -1 in places where the 
        mask does not cover. It will also return the index relative to just the masked region.'''
        cropped_size = np.prod(self.data[xpos:xposend, ypos:yposend, zpos:zposend].shape[:3])
        # self.cropped_indices_map = np.zeros_like(cropped_size)
        # self.cropped_indices_map.fill(-1)
        self.cropped_indices_map = np.zeros(cropped_size).reshape(tmpmask.shape)
        self.roi_cropped_indices_map = np.zeros(cropped_size).reshape(tmpmask.shape)
        # self.cropped_indices_map = self.cropped_indices_map.reshape(tmpmask.shape)
        ''' cropped_indices_map will return '''

        self.cropped_indices_nopad_map = np.zeros_like(origmask)
        self.cropped_indices_nopad_map = self.cropped_indices_map[self.pad:(-1 * self.pad),
            self.pad:(-1 * self.pad) , self.pad:(-1 * self.pad)]
        self.cropped_indices_nopad_map = self.cropped_indices_nopad_map.ravel().astype(np.int16)
        self.cropped_indices_map = self.cropped_indices_map.ravel()

        ''' put -1 where there are no valid pixels '''
        if self.masklabel:
            self.cropped_indices_nopad_map[(origmask.ravel() == 0) | (origlabel.ravel() == 4)] = -1
            self.cropped_indices_map.ravel()[(tmpmask.ravel() == 0) | (tmplabel.ravel() == 4)] = -1
            self.roi_cropped_indices_map.ravel()[(self.roimask.ravel() == 0) | (tmplabel.ravel() == 4)] = -1
        else:
            self.cropped_indices_nopad_map[(origmask.ravel() == 0)] = -1
            self.cropped_indices_map.ravel()[(tmpmask.ravel() == 0)] = -1
            self.roi_cropped_indices_map.ravel()[(self.roimask.ravel() == 0)] = -1

        self.cropped_indices_map[self.cropped_indices_map > -1] = \
            np.arange(0, len(self.cropped_indices_map[self.cropped_indices_map > -1]))
        self.cropped_indices_map = self.cropped_indices_map.astype(np.int16)

        self.roi_cropped_indices_map[self.roi_cropped_indices_map > -1] = \
            np.arange(0, len(self.roi_cropped_indices_map[self.roi_cropped_indices_map > -1]))
        self.roi_cropped_indices_map = self.roi_cropped_indices_map.astype(np.int16)
        # self.cropped_indices_map.fill(-1)
        # self.cropped_indices_map[self.cropped_indices_nopad] = np.arange(0,len(self.cropped_indices_nopad))
        # if self.singleslice:
        #     if self.masklabel:
        #         self.cropped_indices = np.where((tmpmask[tmpmask>0].ravel() > 0) & (tmplabel[tmpmask>0].ravel() != 4))[0]
        #     else:
        #         self.cropped_indices = np.where((tmpmask[tmpmask > 0].ravel() > 0) )[0]
        #     cropped_size = np.prod(self.data[xpos:xposend, ypos:yposend, 0].shape[:2])
        #     self.cropped_indices_map = np.zeros(cropped_size)
        #     self.cropped_indices_map.fill(-1)
        #     self.cropped_indices_map[self.cropped_indices] = np.arange(0, len(self.cropped_indices))

        # assert np.sum(self.mask) == np.sum(tmpmask)

        # nonzeros = len(self.indices)

        # self.X5 = self.X5[0:nonzeros]
        # self.neighbors = self.neighbors[0:nonzeros]
        # self.neighbors_z = self.neighbors_z[0:nonzeros]
        # self.neighbors_y = self.neighbors_y[0:nonzeros]


        if self.fnoutput:
            data = {'data':self.data[xpos:xposend, ypos:yposend, zpos:zposend],
                    'targets': tmplabel,
                    'indices': self.indices,
                    'cropped_indices': self.cropped_indices, # bigger area
                    'cropped_indices_nopad': self.cropped_indices_nopad,  # area with no padding, original area
                    'cropped_indices_map': self.cropped_indices_map,
                    'roi_cropped_indices_map': self.roi_cropped_indices_map,
                    'origsize': self.dataOrigShape,
                    'cropped_size' : self.data[xpos:xposend, ypos:yposend, zpos:zposend].shape[:3],
                    'bounds': [[self.xpos, self.xpos_end],
                               [self.ypos, self.ypos_end],
                               [self.zpos, self.zpos_end]],
                    'datasetNum' : self.dataNum
                    }
            if self.edgemap:
                data.update({'edgemap':self.edgemap[xpos:xposend, ypos:yposend, zpos:zposend]})
            if self.pkl:
                file_obj = open(self.fnoutput+'.obj', 'wb')
                pickle.dump(data, file_obj, protocol=4)
            else:
                with h5py.File(self.fnoutput + '.h5', "w") as f:
                    f.create_dataset('data', data=data['data'], chunks=True)
                    f.create_dataset('targets', data=tmplabel, chunks=True)
                    f.create_dataset('indices', data=self.indices)
                    f.create_dataset('cropped_indices', data=self.cropped_indices)
                    f.create_dataset('cropped_indices_map',data=self.cropped_indices_map)
                    f.create_dataset('cropped_size', data['data'].shape[:3])
                    f.attrs['bounds'] = data['bounds']
                    f.attrs['datasetNum'] = self.dataNum
                    f.attrs['origsize'] = self.dataOrigShape

        print('Dataset files generated.')