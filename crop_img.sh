#!/usr/bin/env bash
# This script crops one (FP) image into many CSxCS crops with overlap (CS>=UCS)
# Typically called by crop_ds.py
# input:
CS=$1       # crop size (including overlap)
UCS=$2      # useful crop size
FP=$3       # input file path
OUTDIR=$4   # directory where results are saved
EXT=${FP,,}
EXT=${EXT:(-3)}
# output:
# OUTDIR/[base filename]_XCROPN_YCROPN_USEFULCROPSIZE.jpg

# TODO note that this script will fail if an image dimension is divisible by UCS

if ! [[ "$CS" =~ ^[0-9]+$ ]] || ((CS%8!=0)) || ! [[ "$UCS" =~ ^[0-9]+$ ]] || ((UCS%8!=0)) 
then
	echo "Syntax: bash $0 [CROPSIZE] [USEFUL CROP SIZE] [FILE PATH] [OUTPUT DIR]"
	echo "Error: ${CS} or ${UCS} is an invalid crop size, must be a multiple of 8."
	exit -1
fi
mkdir -p ${OUTDIR}
NTHREADS=$(grep -c ^processor /proc/cpuinfo)
echo "Cropping ${FP}..."
RES=($(file ${FP} | grep -o -E '[0-9]{3,}( )*x( )*[0-9]{3,}' | tail -1 | grep -o -E '[0-9]+'))
BN=$(basename $FP);BN=${BN::-4}
mkdir -p "${OUTDIR}"
NXCROPS=$((${RES[0]}/$UCS+1))
NYCROPS=$((${RES[1]}/$UCS+1))
let CURX=CURY=0

while ((${CURY}<${NYCROPS}))
do
    # base cases
    let XCS=YCS=CS
    CUCS=$UCS # current useful crop size
    XBEG=$((CURX*UCS-(CS-UCS)/2))
    YBEG=$((CURY*UCS-(CS-UCS)/2))
    # starting from X=0 or Y=0
    if [ $CURX == 0 ]
    then
        XCS=$(($CS-($CS-$UCS)/2))
        XBEG=0
    fi
    if [ $CURY == 0 ]
    then
        YCS=$((CS-(CS-UCS)/2))
        YBEG=0
    fi
    # starting close to the end?
    XCS=$((XCS<${RES[0]}-XBEG?XCS:${RES[0]}-XBEG))
    YCS=$((YCS<${RES[1]}-YBEG?YCS:${RES[1]}-YBEG))
    # starting from Xlast or Ylast
    if [ $CURX == $(($NXCROPS-1)) ]
    then
        CUCS=$(($XCS-($CS-$UCS)/2))
    fi
    if [ $CURY == $(($NYCROPS-1)) ]
    then
        CUCS=$(($CUCS<($YCS-($CS-$UCS)/2)?$CUCS:($YCS-($CS-$UCS)/2)))
    fi
    CPATH="${OUTDIR}/${BN}_${CURX}_${CURY}_${CUCS}.${EXT}"
	if [ ! -f "${CPATH}" ]
	then
		if [ "$EXT" = "jpg" ]; then
			CMD="jpegtran -crop ${XCS}x${YCS}+${XBEG}+${YBEG} -copy none -optimize -outfile ${CPATH} ${FP}"
		else
			CMD="convert -crop ${XCS}x${YCS}+${XBEG}+${YBEG} ${FP} ${CPATH}"
        	fi
		if ! $(${CMD})
		then
		    echo "${CMD}"
	    fi
	fi
	((CURX++))
	if ((CURX>=NXCROPS))
	then
	    CURX=0
	    ((CURY++))
	fi
done
