
import os
import numpy as np
import scipy as sp
import geopandas as gpd
import rasterio
import rioxarray as rxr
from skimage import morphology, filters
import yaml
import warnings
import colorcet as cc
import ipywidgets as wgt
from ipywidgets import Layout
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
from src.phasepack import phasesym, phasecongmono, phasecong, ppdrc, highpassmonogenic
# from src.phasepack import phasesym, phasecongmono, phasecong, highpassmonogenic
from src.lineament_extraction import (
    upcontinue, tilt_angle, clear_short_binary_lines, binarr2lines, merge_connected_lines,
    LineamentDetection
)


class Lineament_Colocation:

    def __init__(self) -> None:

        self.input_parms = None

        self.ws_folder = wgt.Text(
            value = os.path.join('conf', 'multscale_colocation_conf.yml'),
            placeholder="Enter input parameter yaml file path",
            description="YAML Conf:",
            layout=Layout(width="60%"),
        )
        self.load_btn = wgt.Button(description="read parameters", button_style="primary")
        self.load_btn.on_click(self.on_load_parm_click)
        self.output1 = wgt.Output(layout={"border": "1px solid black"})

        self._load = self.load_parm()

        self.rioarr = None
        self.rioarr_loaded = False
        self.show_btn = wgt.Button(description="show input data", button_style="info")
        self.show_btn.on_click(self.show_sources)

        self.transform_method = wgt.Dropdown(
            options=['phasesym', 'tilt'],
                value='phasesym',
                description='transform method:',
                disabled=False,
            )
        self.detection_type =  wgt.Dropdown(
            options=['ridges', 'canny', 'max'],
                value='ridges',
                description='detection type:',
                disabled=False,
            )
        self.apply_wt = wgt.Checkbox(
            value=False,
            description="apply weight",
            disabled=False,
            indent=False
        )

        self.val_mask = None

        self.uc_data = []
        self.transform_btn = wgt.Button(description="Prepare Data", button_style="info")
        self.transform_btn.on_click(self.on_transform_click)
        self.transformed = False

        self.xian = []
        self.detection_btn = wgt.Button(description="Detection", button_style="info")
        self.detection_btn.on_click(self.on_detection_click)
        self.detedted = False

        self.coloc_data = []
        self.colocation_btn = wgt.Button(description="Colocation", button_style="primary")
        self.colocation_btn.on_click(self.on_colocation_click)


        self.out_name = wgt.Text(
            value="",
            placeholder="output name",
            description="raster output",
            layout=Layout(width="30%"),
        )
        self.export_btn = wgt.Button(description="Export", button_style="info")
        self.export_btn.on_click(self.on_export_click)

        self.line_px = None # final pixel lineament binary data

        self.output2 = wgt.Output(layout={"border": "1px solid black"})


    def load_parm(self):
        try:
            label_html = HTML(
                '<p style="font-family: Bond; font-size: 20px;">Input Parameters:</p>'
            )
            display(label_html)
            hb = wgt.VBox([wgt.HBox([self.ws_folder, wgt.HTML(value="&rArr;"), self.load_btn]), self.output1])
            display(hb)

        except Exception as e:
            with self.output1s:
                self.output1.clear_output(wait=True)
                print(f"An unexpected error occured: {e}")       



    def on_load_parm_click(self, btn=None):
        with self.output1:
            self.output1.clear_output(wait=True)
            try:
                pf = self.ws_folder.value

                print(pf)
                if not os.path.exists(pf):
                    raise ValueError("parmaeter file not found!")
        
                with open(pf, 'r') as p:
                    self.input_parms = yaml.safe_load(p)
        
                display(self.input_parms)

            except ValueError as v:
                with self.output1:
                    self.output1.clear_output(wait=True)
                    print(v)

            except Exception as e:
                with self.output1:
                    self.output1.clear_output(wait=True)
                    print(f"An unexpected error occured: {e}")


    def extract(self):

        clear_output()

        display(
            wgt.VBox(
                [
                    self.show_btn,
                    self.transform_method,
                    self.detection_type,
                    self.apply_wt,
                    wgt.HBox(
                        [
                            self.transform_btn,
                            wgt.HTML(value="&rArr;"),
                            self.detection_btn,
                            wgt.HTML(value="&rArr;"),
                            self.colocation_btn
                        ]
                    ),
                    wgt.HBox(
                        [
                            self.out_name,
                            self.export_btn
                        ]
                    ),
                    self.output2,
                ]
            )
        )
    def _replace_nodata_with_nan(self, din):
        nodata = float(self.input_parms['parms']['nodata'])
        din[din==nodata] = np.nan
        return din

    def show_sources(self, btn=None):
        with self.output2:
            self.output2.clear_output(wait=True)
            infile = self.input_parms['inputs']['file path']

            if not self.rioarr_loaded:
                self.rioarr = rxr.open_rasterio(infile)
            else:
                self.rioarr_loaded = True
            shape = self.rioarr.data.shape
            if len(shape) ==2:
                h, w = shape
            elif len(shape) == 3:
                shape2 = [i for i in shape if i!=min(shape)]
                h, w = shape2

            fig, ax = plt.subplots(1, 2, figsize=(2*round(h/w*6), 6))
            self.rioarr.plot(ax=ax[0])
            ax[0].tick_params(axis='x', rotation=45)
            ax[0].set_aspect('equal')
            self.rioarr.plot.hist(ax=ax[1])
            plt.show()

    def on_transform_click(self, btn=None):

        with self.output2:
            self.output2.clear_output(wait=True)
            self.detected = False
            if not self.rioarr_loaded:
                self.rioarr = rxr.open_rasterio(self.input_parms['inputs']['file path'])
            else:
                self.rioarr_loaded = True

            if self.rioarr.rio.count != 1:
                raise ValueError('raster has more than one band, only single band raster allowed!')

            # check resolution            
            if np.mean(np.absolute(self.rioarr.rio.resolution()))!=self.input_parms['parms']['spacing'] and \
                not not self.input_parms['parms']['use input spacing']:
                print(f"res - {self.rioarr.rio.resolution(self.input_parms['parms']['spacing'])}")
                raise ValueError('The input spacing might be different from the source data resolution!')
            elif self.input_parms['parms']['use input spacing']:
                print(f"using input raster resolution {self.input_parms['parms']['spacing']}")
            

            data_in = np.squeeze(self.rioarr.data)
            if self.rioarr.rio.nodata != float(self.input_parms['parms']['nodata']):
                warnings.warn(f"detected nodata {self.rioarr.rio.nodata} != input nodata {self.input_parms['parms']['nodata']}!")
            data_in = self._replace_nodata_with_nan(data_in)
            
            # fill nan
            if np.any(np.isnan(data_in)):
                self.val_mask = np.where(~np.isnan(data_in))
                interp = sp.interpolate.NearestNDInterpolator(np.transpose(self.val_mask), data_in[self.val_mask])
                data_in = interp(*np.indices(data_in.shape))
                
            # filter parameter affects the outcomes, so need to experiment
            data_in = filters.butterworth(data_in, 0.005, True)
            # transform
            cols, rows = np.meshgrid(np.arange(self.rioarr.rio.width), np.arange(self.rioarr.rio.height))
            ds = self.input_parms['parms']['spacing']
            trans_method=self.transform_method.value
            print('Preparing...')
            for h in self.input_parms['parms']['uc height']:
                dd = upcontinue(cols*ds, rows*ds, data_in, data_in.shape, h).reshape(data_in.shape)
                dd = filters.difference_of_gaussians(dd, 1.5)
                self.uc_data.append(dd)

                if trans_method=='phasesym':
                    this_xian, orient, symmetryEnergy, T = phasesym(dd, nscale=4, norient=8, minWaveLength=3, mult=2.1, sigmaOnf=0.25, k=2, polarity=0, noiseMethod=-1)
                elif trans_method=='tilt':
                    in_x, in_y= np.meshgrid(self.rioarr.x.values, self.rioarr.y.values)
                    this_xian = tilt_angle(in_x[-1], in_y[:, 0], dd.ravel(), dd.shape, sigma=1).reshape(dd.shape)
                self.xian.append(this_xian)
            self.transformed = True
            print('Done...')
                
            if True: # do plot
                N = int(np.ceil(len(self.input_parms['parms']['uc height'])*0.5))
                fig, ax = plt.subplots(2, N, figsize=(24, 10))
                ax = ax.ravel()
                for i, h in enumerate(self.input_parms['parms']['uc height']):
                    ax[i].imshow(self.xian[i], cmap=cc.cm.coolwarm)
                    ax[i].set_title(str(h)+'m')
                plt.show()



    def on_detection_click(self, btn=None):
        with self.output2:
            self.output2.clear_output(wait=True)

            if not self.transformed:
                raise ValueError('Run preparation first!')
            print('Detecting ...')
            detect_type=self.detection_type.value
            if detect_type == 'ridges': # sato filter
                parm = {'black_ridges': False, 'filter_obj_size': 5, 'block_size': None}
            elif detect_type == 'canny': 
                parm = {'sigma': 3}
            elif detect_type =='max':  # max
                parm = {'threshold_type': 'global', 'global_threshold': None, 'local_block_size': 'auto'}
            else:
                print("wrong type name!")
            

            lineobj = LineamentDetection()
            heights = self.input_parms['parms']['uc height']
            for this_xian, h in zip(self.xian, heights):
                # if h>8000: # apply a taper to reduce the edge effects
                #     this_xian *=filters.window(('tukey', 0.3), this_xian.shape)
                self.coloc_data.append(lineobj.fit(this_xian, affine=self.rioarr.rio.transform()).transform(type=detect_type, **parm))
            self.detected= True
            print('Done...')
            if True: # do plot
                N = int(np.ceil(len(self.input_parms['parms']['uc height'])*0.5))
                fig, ax = plt.subplots(2, N, figsize=(24, 10))
                ax = ax.ravel()
                for i, h in enumerate(heights):
                    ax[i].imshow(self.coloc_data[i], cmap=cc.cm.dimgray_r)
                    ax[i].set_title(str(h)+'m')
                plt.show()
                    
    def on_colocation_click(self, btn=None):
        with self.output2:
            self.output2.clear_output(wait=True)
            print('Colocation...')
            if not self.detected:
                raise ValueError('detection needs to run!')
            apply_wt = self.apply_wt.value
            buffer=5
            heights = self.input_parms['parms']['uc height']
            line_px = 1*morphology.dilation(self.coloc_data[0], morphology.square(3)) * (heights[0]*0.001 if apply_wt else 1.)
            for i, h in zip(self.coloc_data[1::], heights[1::]):
                line_px += morphology.dilation(i, morphology.square(buffer))*(h*0.001 if apply_wt else 1.)
            line_px2 = np.full(line_px.shape, np.nan)
            line_px2[self.val_mask] = line_px[self.val_mask]
            del line_px

            self.line_px = self.rioarr.copy()
            self.line_px.data = line_px2.reshape(self.rioarr.data.shape)

                
            # display
            if True:
                fig, ax = plt.subplots(1, 2, figsize=(24, 12))
                self.line_px.plot(cmap = cc.cm.dimgray_r, ax=ax[0])
                # ax1=ax[0].imshow(self.line_px, extent=self.rioarr.rio.bounds(), cmap=cc.cm.dimgray_r)
                # ax[0].axis('equal')
                # fig.colorbar(ax1,ax=ax[0], shrink=0.9)
                self.rioarr.plot(cmap=cc.cm.coolwarm, ax=ax[1])
                # ax2 = ax[1].imshow(data_og, extent=extent_og, cmap=cc.cm.coolwarm) 
                # fig.colorbar(ax2,ax=ax[1], shrink=0.6)
                plt.show()

    def on_export_click(self, btn=None):

        with self.output2:
            self.output2.clear_output(wait=True)
            out_name = os.path.join(self.input_parms['outputs']['folder'], self.out_name.value)
            _, ext = os.path.splitext(out_name)
            if ext != 'tif' or ext != 'tiff':
                out_name += '.tif'
            self.line_px.rio.to_raster(
                out_name,
                compress="LZW",
                nodata=np.nan,
                driver="GTiff",
                description = "uc colocation lineament extraction."
            )
            


            print(f'saved to ... {out_name}')




class Raster_To_Lines():

    def __init__(self) -> None:
        
        self.inarr = None        
        self.lines_gpd = None

        self.raster_file = wgt.Text(
                    value=os.getcwd(),
                    placeholder="Enter raster file path",
                    description="File path:",
                    layout=Layout(width="60%"),
                )

        self.nodata = wgt.Text(
                    value='0',
                    description="NODATA:",
                    layout=Layout(width="20%"),
        )
        self.threshold = wgt.Text(
                    value='0',
                    description="threshold:",
                    layout=Layout(width="20%"),
        )
        self.eps = 0.1
        self.min_pixel_cluster = 5

        self.convert_btn = wgt.Button(description="Convert to lineaments", button_style="primary")
        self.convert_btn.on_click(self.on_convert_click)

        self.output_folder = wgt.Text(
                    placeholder="Enter output folder path:",
                    description="folder path",
                    layout=Layout(width="60%"),
                )
        self.out_name = wgt.Text(
                    placeholder="output file name:",
                    description="Output name:",
                    layout=Layout(width="60%"),
                )
        self.save = wgt.Button(description="Save", button_style="primary")
        self.save.on_click(self.on_save_click)


        self.output = wgt.Output(layout={"border": "1px solid black"})
        display(
            wgt.VBox(
                [self.raster_file,
                 self.nodata,
                 self.threshold,
                 self.convert_btn,
                 self.output_folder,
                 self.out_name,
                 self.save,
                 self.output]
            )
        )


    # def raster_to_lines_wrapper(tif_path, nodata, binary_threshold, min_pixels, eps):
    def on_convert_click(self, btn=None):
        """_summary_

        Args:
            tif_path (_type_): _description_
            nodata (_type_): _description_
            binary_threshold (_type_): _description_
            min_pixels (_type_): _description_
            eps (_type_): _description_

        Raises:
            ValueError: _description_
        """
        with self.output:
            self.output.clear_output(wait=True)
            print('converting...')
            if not os.path.exists(self.raster_file.value):
                raise ValueError('check input file path!')

            self.inarr = rxr.open_rasterio(self.raster_file.value)

            if self.inarr.rio.count != 1:
                raise ValueError('raster has more than one band, only single band raster allowed!')
            din = np.squeeze(self.inarr.data)
            nodata = self.nodata.value
            if len(nodata)>0:
                nodata = float(nodata)
                din[din==nodata] = np.nan
            binary_threshold = float(self.threshold.value)
            din = np.nan_to_num(din, nan=binary_threshold)
            coloc = din>binary_threshold
            coloc = clear_short_binary_lines(coloc, self.min_pixel_cluster)

            coloc_lines, coloc_intxn_xy = binarr2lines(morphology.thin(coloc), affine=self.inarr.rio.transform())
            coloc_lines = merge_connected_lines(coloc_lines, coloc_intxn_xy, eps = self.eps)

            # pot_lines2 = filter_line_by_curvature(pot_lines)
            len_scale = 6373 if self.inarr.rio.crs.is_geographic else 1

            self.lines_gpd = gpd.GeoDataFrame({'geometry': coloc_lines, 'length': [i.length*len_scale for i in coloc_lines]}, crs=self.inarr.rio.crs)

            print('done ...')
            self.lines_gpd.plot(figsize=(12, 8))
            plt.show()

    def on_save_click(self, btn=None):

        with self.output:
            self.output.clear_output(wait=True)
            out_folder = self.output_folder.value
            if not os.path.exists(out_folder):
                raise ValueError('output folder not found!')
            out_name = self.out_name.value
            _, ext = os.path.splitext(out_name)
            if ext != 'shp':
                out_name += '.shp'

            out_path = os.path.join(out_folder, out_name)
            self.lines_gpd.to_file(out_path)
            print(f'saved to {out_path}')
            