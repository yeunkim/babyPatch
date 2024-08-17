import nibabel as nib
import numpy as np
from normalization import normalize_mri
from scipy import spatial

class imagepatches(object):

    def __init__(self, fname, mask, label,
                 gm=150, wm=250, csf=10, num_classes=4, pad = 3, masklabel=False,
                 fname_t1=None, k_t2=4, k_t2_init=None,
                 k_t1=2, k_t1_init=None, threedim=False, channels=None, coords=None,
                 origcoords = None):

        self.pad = pad
        if not coords is None:
            self.coords = nib.load(coords).get_data()
        else:
            self.coords = coords
        self.origcoords = origcoords

        self.masklabel = masklabel

        self.fname = fname
        self.fname_t1 = fname_t1
        self.fname_mask = mask

        self.k_t2 = k_t2
        self.k_t1 = k_t1
        self.k_t2_init=k_t2_init
        if k_t2_init is None:
            self.k_t2_init = [500,100,1000,1500]
        self.k_t1_init = k_t1_init
        if k_t1_init is None:
            self.k_t1_init = [500,250]

        self.nii = nib.load(fname)
        if self.fname_t1:
            self.nii_t1 = nib.load(fname_t1)
            self.data_t1 = self.nii_t1.get_data()
        self.data = self.nii.get_data()
        self.dataOrigShape = self.data.shape

        self.indices = []
        self.indices_upsampled = []

        self.nii3 = nib.load(mask)
        self.mask = self.nii3.get_data()

        self.gm = gm
        self.wm = wm
        self.csf = csf
        self.label = nib.load(label).get_data()

        self.label[self.label == 255] = 1
        self.label[self.label == 0] = 4
        self.label[self.label == wm] = 0
        self.label[self.label == gm] = 1
        self.label[self.label == csf] = 2
        self.num_classes = num_classes

        self.indices_upsampled =[]

        self.upsample()
        self.get_bounds()

        if threedim:
            self.normalize()
            self.render_patches_3d()
            self.create_data_struct_3d()
        elif channels:
            self.channels = channels
            self.render_patches_channels()
            self.create_data_struct()
        elif not self.origcoords is None:
            self.normalize()
            self.render_patches_simcoords()
            self.create_data_struct()
        else:
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
            if np.sum(self.mask[i, :,:]) > 0:
                self.xpos = i
                break

        for i in range(self.mask.shape[0]-1, 0, -1):
            if np.sum(self.mask[i, :,:]) > 0:
                self.xpos_end = i-1
                break

        for i in range(self.mask.shape[1]):
            if np.sum(self.mask[:, i,:]) > 0:
                self.ypos = i
                break

        for i in range(self.mask.shape[1]-1, 0, -1):
            if np.sum(self.mask[:, i,:]) > 0:
                self.ypos_end = i-1
                break

        for i in range(self.mask.shape[2]):
            if np.sum(self.mask[:, :,i]) > 0:
                self.zpos = i
                break

        for i in range(self.mask.shape[2]-1, 0, -1):
            if np.sum(self.mask[:, :,i]) > 0:
                self.zpos_end = i-1
                break

    def render_patches(self):
        pad = self.pad
        ## break down into patches
        # approximate array allocation
        N = int(np.count_nonzero(self.mask))
        self.X = np.zeros([N,1])
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

        for k in range(self.zpos, self.zpos_end):
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
                    j += 1
                i +=1

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
        self.neighbors = self.neighbors[0:nonzeros]
        self.neighbors_z = self.neighbors_z[0:nonzeros]
        self.neighbors_y = self.neighbors_y[0:nonzeros]
        # self.line = self.line[0:nonzeros,:]
        # self.zline = self.zline[0:nonzeros, :]
        if not self.coords is None:
            self.coordsvec = self.coordsvec[0:nonzeros]

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