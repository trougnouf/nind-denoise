DESTDIR=/orb/tmp/cleanrepo
ORIGDIR="$(pwd)/../.."
mkdir -p ${DESTDIR}
cd ${DESTDIR}
#git clone git@github.com:trougnouf/nind-denoise.git
DESTDIR=${DESTDIR}/nind-denoise
cd ${DESTDIR}
git pull
rm * -r
CURDIR="models/nind_denoise/2019-02-18T20:10_run_nn.py_--time_limit_259200_--batch_size_94_--test_reserve_ursulines-red_stefantiek_ursulines-building_MuseeL-Bobo_CourtineDeVillersDebris_MuseeL-Bobo-C500D_--skip_sizecheck_--lr_3e-4"
mkdir -p ${CURDIR}
cp ${ORIGDIR}/${CURDIR}/model* "${CURDIR}"
CURDIR="models/nind_denoise/2019-08-03T16:14_nn_train.py_--g_network_UNet_--weight_SSIM_1_--batch_size_65_--test_reserve_ursulines-red_stefantiek_ursulines-building_MuseeL-Bobo_CourtineDeVillersDebris_MuseeL-Bobo-C500D_--train_data_datasets-train-NIND_128_96_--g_model_path_models-20"
mkdir -p ${CURDIR}
cp ${ORIGDIR}/${CURDIR}/generator* "${CURDIR}"
mkdir -p src/common/libs
cp ${ORIGDIR}/src/common/freelibs/*.py src/common/libs/
mkdir -p src/nind_denoise/configs
cp ${ORIGDIR}/src/nind_denoise/configs/* src/nind_denoise/configs/
mkdir -p src/nind_denoise/networks
cp ${ORIGDIR}/src/nind_denoise/networks/Hul.py ${ORIGDIR}/src/nind_denoise/networks/nnModules.py ${ORIGDIR}/src/nind_denoise/networks/p2p_networks.py ${ORIGDIR}/src/nind_denoise/networks/relics.py ${ORIGDIR}/src/nind_denoise/networks/ThirdPartyNets.py ${ORIGDIR}/src/nind_denoise/networks/UtNet.py src/nind_denoise/networks/
mkdir -p src/nind_denoise/deprecated
cp ${ORIGDIR}/src/nind_denoise/deprecated/* src/nind_denoise/deprecated/
mkdir -p src/nind_denoise/libs
cp ${ORIGDIR}/src/nind_denoise/libs/pytorch_ssim/ ${ORIGDIR}/src/nind_denoise/libs/graph_utils.py src/nind_denoise/libs/ -r
mkdir -p src/nind_denoise/tools
cp ${ORIGDIR}/src/nind_denoise/tools/* ${ORIGDIR}/src/nind_denoise/libs/graph_utils.py src/nind_denoise/tools/
cp -r "${ORIGDIR}/src/nind_denoise/unittest_resources" ${ORIGDIR}/src/nind_denoise/*.py ${ORIGDIR}/src/nind_denoise/README.md src/nind_denoise/
cp -r ${ORIGDIR}/src/nind_denoise/datasets src/nind_denoise/
cp src/nind_denoise/README.md .
echo "Clean repo ready in $DESTDIR"