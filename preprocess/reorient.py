from PIL import Image
import numpy as np
import nibabel as nib
import os
import glob

# make tiff into nifti
# pdir = '/data/mouse/CHLA/Train12/'
# pdir = '/data/mouse/CHLA/StdRes_MRI_Data/'
pdir = '/data/mouse/Neil_TBI/mouseTest16/'
# pdir = '/data/rat/Neil_TBI/Test12/'
# ids = ["29312_06222021",
# "29322_07132021",
# "29332_07142021",
# "29358_06222021",
# "29359_06222021",
# "29365_06292021",
# "9201_07292021",
# "9202_07292021",
# "9243_07142021",
# "9244_07152021",
# "9254_06282021",
# "9261_06252021"]
ids = []
# with open('/data/mouse/CHLA/StdRes_MRI_Data/subjs.txt', 'r') as f:
with open('/data/mouse/Neil_TBI/mouseTestIDs.txt', 'r') as f:
# with open('/data/rat/Neil_TBI/testIDs.txt') as f:
    subjids = f.readlines()
    for id in subjids:
        ids.append((id.rstrip().lstrip()))
# for id in ids:
#     print(id)
#     try:
#         nii = nib.load(pdir + id + '/{0}.nii.gz'.format(id))
#         mri = np.zeros((320,320,48), dtype=np.int32)
#         basename = os.path.basename(glob.glob(pdir + id + os.sep + "label/*.tif")[0]).split('.')[0]
#         for i in range(0, 48):
#             im = Image.open(pdir + id + '/label/{1}.labels{0}.tif'.format(str(i).zfill(2), basename))
#             mri[..., i] = np.array(im)
#
#         recon = nib.Nifti1Image(mri, affine=nii._affine, header=nii.header)
#
#         nib.save(recon, pdir + id + '/brain.label.nii.gz')
#     except FileNotFoundError:
#         print('No nifti')
#         continue
#     except IndexError:
#         print('No label found')
#         continue

# ids = ['9203_07292021','29315_06212021']

# nii = nib.load('/Users/yeunkim/Downloads/29358_06222021_T2_TurboRARE_highresLevMoatsV2_20210622133656_20001.nii.gz')
# path = '/data/mouse/Neil_TBI/mouseTrain16/'
# ids = ["d000_inj_pig_f_08.2",
# "d000_inj_veh_m_15.3",
# "d000_shm_pig_f_02.1",
# "d000_shm_pig_m_12.2",
# "d030_inj_veh_f_06.2",
# "d030_inj_pig_m_05.3",
# "d030_shm_pig_f_10.2",
# "d030_shm_pig_m_12.3",
# "d100_inj_pig_f_14.3",
# "d100_inj_veh_m_04.2",
# "d100_shm_veh_f_09.1",
# "d100_shm_pig_m_03.4",
# "d166_inj_pig_f_06.3",
# "d166_inj_pig_m_07.1",
# "d166_shm_veh_f_09.3",
# "d166_shm_veh_m_11.1"]

# ids = ['29366_06292021']
# ids = []
# with open('/data/mouse/CHLA/StdRes_MRI_Data/nii_and_label.txt', 'r') as f:
#     subjids = f.readlines()
#     for id in subjids:
#         ids.append((id.rstrip().lstrip()))
for id in ids:

    # # nii = nib.load(pdir + id + '/{0}.nii.gz'.format(id))
    # nii = nib.load(pdir + id + '/anat.nii.gz')
    # # nii = nib.load(pdir + id + '/T2star_MEAN.nii.gz'.format(id))
    # # nii = nib.load(pdir + id + '/bet_mask.nii.gz')
    # # # orientation for chla
    # # ornt = np.array([[0, 1],
    # #                  [1, -1],
    # #                  [2, 1]])
    #
    # ornt = np.array([[0, 1],
    #                  [1, 1],
    #                  [2, 1]])
    #
    # img_orient = nii.as_reoriented(ornt)
    #
    # orig_img = img_orient.get_fdata()
    #
    # # neil rat
    # # orig_img_ = np.moveaxis(orig_img, 0, 2)
    # # orig_img_ = np.moveaxis(orig_img_, 2, 1)
    # orig_img_ = np.moveaxis(orig_img, 2, 1) # neil mouse
    # # orig_img_ = np.moveaxis(orig_img_, 0, 1)
    #
    # x = nii._affine[0]
    # y = nii._affine[2]
    # z = nii._affine[1]
    # affine = np.zeros((4,4))
    # # rat neil
    # # affine[0] = x
    # # affine[1] = y
    # # affine[2] = z
    # # affine[-1,-1] = 1
    # affine[0,0] = x[0]
    # affine[1,1] = y[2]
    # affine[2,2] = z[1]
    # affine[:,-1] = nii._affine[:,-1]
    #
    # # orig_img_ = np.flip(orig_img_, 2)
    # orig_img_data_ = nib.Nifti1Image(orig_img_, affine)
    #
    # nib.save(orig_img_data_, pdir + id + '/anat.reorient.nii.gz')
    # # nib.save(orig_img_data_, pdir + id + '/brain.reorient.label.nii.gz')



    ############# skull stripped data

    # nii = nib.load('/data/mouse/CHLA/Train/29360_06222021/skullstripped.nii.gz')
    # nii = nib.load(pdir + id + '/brain.label.nii.gz')
    nii = nib.load(pdir + id + '/bet_mask.nii.gz')
    # chla
    # ornt = np.array([[0, 1],
    #                  [1, 1],
    #                  [2, 1]])
    # neil
    ornt = np.array([[0, -1],
                    [1, 1],
                    [2, 1]])

    img_orient = nii.as_reoriented(ornt)

    orig_img = img_orient.get_fdata()

    # orig_img_ = np.moveaxis(orig_img, 0, 2)
    orig_img_ = np.moveaxis(orig_img, 2, 1)

    x = nii._affine[0]
    y = nii._affine[2]
    z = nii._affine[1]
    affine = np.zeros((4,4))
    affine[0,0] = x[0]
    affine[1,1] = y[2]
    affine[2,2] = z[1]
    affine[:,-1] = nii._affine[:,-1]
    orig_img_data_ = nib.Nifti1Image(orig_img_, affine)

    nib.save(orig_img_data_, pdir + id + '/brain.reorient.label.nii.gz')