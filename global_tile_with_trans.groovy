import qupath.lib.roi.*
import java.awt.geom.AffineTransform
import qupath.lib.roi.interfaces.ROI
import java.lang.Math

// Write the full image (only possible if it isn't too large!)
def server = getCurrentServer()
//writeImage(server, '/path/to/export/full.tif')
def wsi_name = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
// Write the full image downsampled by a factor of 10
def requestFull = RegionRequest.createInstance(server, 10)
//writeImageRegion(server, requestFull, '/path/to/export/full_downsampled.tif')

// Write the region of the image corresponding to the currently-selected object
def (X0, Y0, X1, Y1) = [102400, 103680, 106496, 99840]

X0 = [X0, X1].min()
Y0 = [Y0, Y1].min()

def downsample = 1.0
double size = 1024

def matrix = [1.000,    -0.0,    -606,
-0.0,    1.0,    1468]
def dx = Math.abs(matrix[2])
def dy = Math.abs(matrix[5])

if (wsi_name.contains("IHC"))
{// matrix for IHC.
// negate the shear of matrix to calibrate HE against IHC
   print("IHC")
}else{
print("HE")
// matrix for HE
matrix = [1.000,    -0.0,    0,
-0.0,    1.0,    0]
//X0 = X1
//Y0 = Y1
}


def transform = new AffineTransform(
        matrix[0], matrix[3], matrix[1],
        matrix[4], matrix[2], matrix[5]
)


def (sx, sy, ex, ey) = [0.0, 0.0, X0, Y0]


sx = 20400
sy = 80400
ex = sx+10000-dx
ey = sy+10000-dy


def sx0, sy0
sx0 = sx + dx
sy0 = sy + dy

def roi
def requestROI
def name
def k = 0
def ext = "tif"

pathToSave = "/Users/cunyuan/DATA/Kimura/qupath-proj/tiles/${wsi_name}_${size}_${downsample}/"

File  f = new File(pathToSave)
f.mkdirs()

size *= downsample
for (sx = sx0; sx < ex - size ; sx += size) {
for (sy = sy0; sy < ey - size ; sy += size) {
    roi = new RectangleROI(sx, sy, size, size)

    roi = transformROI(roi, transform)
    print(k)
    requestROI = RegionRequest.createInstance(server.getPath(), downsample, roi)
    print(roi)
    //name = '%s (d=%.2f, x=%d, y=%d, w=%d, h=%d, z=%d).%s'%(wsi_name, downsample, sx, sy, size, size, k, ext)
    name = "${pathToSave}${k}_${wsi_name} (d=$downsample, x=$sx, y=$sy, w=$size, h=$size, z= ${k}).tif".toString()
    print(name)
    writeImageRegion(server, requestROI, name)
   
    k++
    }
}

print('[DONE]')

/**
 * Transform ROI (via conversion to Java AWT shape)
 *
 * @param roi
 * @param transform
 * @return
 */
ROI transformROI(ROI roi, AffineTransform transform) {
    def shape = RoiTools.getShape(roi) // Should be able to use roi.getShape() - but there's currently a bug in it for rectangles/ellipses!
    shape2 = transform.createTransformedShape(shape)
    return RoiTools.getShapeROI(shape2, roi.getImagePlane(), 0.5)
}
