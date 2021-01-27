//https://forum.image.sc/t/exporting-polygons-of-cell-detections-objects-along-in-csv/44959

import static qupath.lib.gui.scripting.QPEx.*
import com.google.gson.GsonBuilder
import qupath.lib.gui.tools.MeasurementExporter

def cells = getCellObjects()

boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)
File file = new File('/Users/cunyuan/7930.json')
file.withWriter('UTF-8') {
    gson.toJson(cells,it)
}
