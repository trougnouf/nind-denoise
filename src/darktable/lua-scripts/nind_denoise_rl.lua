--[[
  darktable is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  darktable is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with darktable.  If not, see <http://www.gnu.org/licenses/>.
]]

--[[
  DESCRIPTION
    nind_denoise_rl.lua - NIND-denoise then Richardson-Lucy output sharpening using GMic

    This script provides a new target storage "NIND-denoise RL".
    Images exported will be denoised with NIND-denoise then sharpened with GMic's RL deblur

  REQUIRED SOFTWARE
    NIND-denoise: https://github.com/trougnouf/nind-denoise
    GMic command line interface (CLI) https://gmic.eu/download.shtml
    exiftool to copy EXIF to the final image

  USAGE
    * start the script "nind_denoise_rl" from Script Manager
    * in lua preferences:
      - paste in the nind_denoise command, including the --model-path, e.g.:
        python3 ~/nind-denoise/src/nind_denoise/denoise_image.py  --model_path "~/nind-denoise/models/2021-06-14T20_27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11_48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w/generator_650.pt"
      - select GMic cli executable (for RL-deblur)
      - select the exiftool cli executable (optional, to copy EXIF to final image)
    * from "export selected", choose "nind-denoise RL" as target storage
    * for "format options", either TIFF 8-bit or 16-bit is recommended
]]

local dt = require "darktable"
local du = require "lib/dtutils"
local df = require "lib/dtutils.file"
local dtsys = require "lib/dtutils.system"

-- module name
local MODULE_NAME = "nind_denoise_rl"

-- check API version
du.check_min_api_version("7.0.0", MODULE_NAME)

-- return data structure for script_manager

local script_data = {}

script_data.destroy = nil -- function to destory the script
script_data.destroy_method = nil -- set to hide for libs since we can't destroy them commpletely yet, otherwise leave as nil
script_data.restart = nil -- how to restart the (lib) script after it's been hidden - i.e. make it visible again

-- OS compatibility
local PS = dt.configuration.running_os == "windows" and  "\\"  or  "/"

-- translation
local gettext = dt.gettext
gettext.bindtextdomain(MODULE_NAME, dt.configuration.config_dir..PS.."lua"..PS.."locale"..PS)
local function _(msgid)
  return gettext.dgettext(MODULE_NAME, msgid)
end

-- initialize module preferences
if not dt.preferences.read(MODULE_NAME, "initialized", "bool") then
  dt.preferences.write(MODULE_NAME, "output_path", "string", "$(FILE_FOLDER)/darktable_exported/$(FILE_NAME)")
  dt.preferences.write(MODULE_NAME, "output_format", "integer", 1)
  dt.preferences.write(MODULE_NAME, "sigma", "string", "1")
  dt.preferences.write(MODULE_NAME, "iterations", "string", "20")
  dt.preferences.write(MODULE_NAME, "jpg_quality", "string", "95")
  dt.preferences.write(MODULE_NAME, "initialized", "bool", true)
end


-- namespace variable
local NDRL = {};

local function denoise_rldeblur_toggled()
  NDRL.sigma_slider.sensitive = NDRL.rl_deblur_chkbox.value
  NDRL.iterations_slider.sensitive = NDRL.rl_deblur_chkbox.value

  -- hide the output format if neither checkboxes selected
  local passthrough = NDRL.rl_deblur_chkbox.value == false and NDRL.denoise_chkbox.value == false
  NDRL.output_format.visible = not passthrough
  NDRL.jpg_quality_slider.visible = not passthrough
end


local function output_format_changed()
  if NDRL.output_format == nil then
    return true
  end

  if NDRL.output_format.selected == 1 then
    NDRL.jpg_quality_slider.visible = true
  else
    NDRL.jpg_quality_slider.visible = false
  end

  dt.preferences.write(MODULE_NAME, "output_format", "integer", NDRL.output_format.selected)
end


-- namespace variable
NDRL = {
  substitutes = {},
  placeholders = {"ROLL_NAME","FILE_FOLDER","FILE_NAME","FILE_EXTENSION","ID","VERSION","SEQUENCE","YEAR","MONTH","DAY",
                  "HOUR","MINUTE","SECOND","EXIF_YEAR","EXIF_MONTH","EXIF_DAY","EXIF_HOUR","EXIF_MINUTE","EXIF_SECOND",
                  "STARS","LABELS","MAKER","MODEL","TITLE","CREATOR","PUBLISHER","RIGHTS","USERNAME","PICTURES_FOLDER",
                  "HOME","DESKTOP","EXIF_ISO","EXIF_EXPOSURE","EXIF_EXPOSURE_BIAS","EXIF_APERTURE","EXIF_FOCUS_DISTANCE",
                  "EXIF_FOCAL_LENGTH","LONGITUDE","LATITUDE","ELEVATION","LENS","DESCRIPTION","EXIF_CROP"},

  output_folder_path = dt.new_widget("entry") {
    tooltip = _("$(ROLL_NAME) - film roll name\n") ..
              _("$(FILE_FOLDER) - image file folder\n") ..
              _("$(FILE_NAME) - image file name\n") ..
              _("$(FILE_EXTENSION) - image file extension\n") ..
              _("$(ID) - image id\n") ..
              _("$(VERSION) - version number\n") ..
              _("$(SEQUENCE) - sequence number of selection\n") ..
              _("$(YEAR) - current year\n") ..
              _("$(MONTH) - current month\n") ..
              _("$(DAY) - current day\n") ..
              _("$(HOUR) - current hour\n") ..
              _("$(MINUTE) - current minute\n") ..
              _("$(SECOND) - current second\n") ..
              _("$(EXIF_YEAR) - EXIF year\n") ..
              _("$(EXIF_MONTH) - EXIF month\n") ..
              _("$(EXIF_DAY) - EXIF day\n") ..
              _("$(EXIF_HOUR) - EXIF hour\n") ..
              _("$(EXIF_MINUTE) - EXIF minute\n") ..
              _("$(EXIF_SECOND) - EXIF seconds\n") ..
              _("$(EXIF_ISO) - EXIF ISO\n") ..
              _("$(EXIF_EXPOSURE) - EXIF exposure\n") ..
              _("$(EXIF_EXPOSURE_BIAS) - EXIF exposure bias\n") ..
              _("$(EXIF_APERTURE) - EXIF aperture\n") ..
              _("$(EXIF_FOCAL_LENGTH) - EXIF focal length\n") ..
              _("$(EXIF_FOCUS_DISTANCE) - EXIF focus distance\n") ..
              _("$(EXIF_CROP) - EXIF crop\n") ..
              _("$(LONGITUDE) - longitude\n") ..
              _("$(LATITUDE) - latitude\n") ..
              _("$(ELEVATION) - elevation\n") ..
              _("$(STARS) - star rating\n") ..
              _("$(LABELS) - color labels\n") ..
              _("$(MAKER) - camera maker\n") ..
              _("$(MODEL) - camera model\n") ..
              _("$(LENS) - lens\n") ..
              _("$(TITLE) - title from metadata\n") ..
              _("$(DESCRIPTION) - description from metadata\n") ..
              _("$(CREATOR) - creator from metadata\n") ..
              _("$(PUBLISHER) - publisher from metadata\n") ..
              _("$(RIGHTS) - rights from metadata\n") ..
              _("$(USERNAME) - username\n") ..
              _("$(PICTURES_FOLDER) - pictures folder\n") ..
              _("$(HOME) - user's home directory\n") ..
              _("$(DESKTOP) - desktop directory"),
    placeholder = _("leave blank to use the location selected below"),
    editable = true,
  },

  output_folder_selector = dt.new_widget("file_chooser_button") {
    title = _("select output folder"),
    tooltip = _("select output folder"),
    value = dt.preferences.read(MODULE_NAME, "output_folder", "string"),
    is_directory = true,
    changed_callback = function(self)
      dt.preferences.write(MODULE_NAME, "output_folder", "string", self.value)
    end
  },

  output_format = dt.new_widget("combobox") {
    label = _("output format"),
    editable = false,
    selected = 1,
    _("JPG"),
    _("TIFF"),
    changed_callback = function(self) output_format_changed() end
  },

  jpg_quality_slider = dt.new_widget("slider") {
    label = _("output jpg quality"),
    tooltip = _("quality of the output jpg file"),
    soft_min = 70,
    soft_max = 100,
    hard_min = 70,
    hard_max = 100,
    step = 2,
    digits = 0,
    value = 95.0,
  },

  denoise_chkbox = dt.new_widget("check_button") {
    label = _("apply nind-denoise"),
    tooltip = _("apply nind-denoise"),
    clicked_callback = function(self) denoise_rldeblur_toggled() end
  },

  rl_deblur_chkbox = dt.new_widget("check_button") {
    label = _("apply RL deblur"),
    tooltip = _("apply GMic's Richardson-Lucy deblur/sharpening"),
    clicked_callback = function(self) denoise_rldeblur_toggled() end
  },

  sigma_slider = dt.new_widget("slider") {
    label = _("sigma"),
    tooltip = _("controls the width of the blur that's applied"),
    soft_min = 0.3,
    soft_max = 2.0,
    hard_min = 0.0,
    hard_max = 3.0,
    step = 0.05,
    digits = 2,
    value = 1.0
  },

  iterations_slider = dt.new_widget("slider") {
    label = _("iterations"),
    tooltip = _("increase for better sharpening, but slower"),
    soft_min = 0,
    soft_max = 100,
    hard_min = 0,
    hard_max = 100,
    step = 5,
    digits = 0,
    value = 10.0
  }
}




-- temp export formats: jpg and tif are supported -----------------------------
local function supported(storage, img_format)
  -- only accept TIF for lossless intermediate file.
  -- JPG compression inteferes with denoising
  return (img_format.extension == "tif")
end



-- shamelessly copied the pattern-replacement functions from rename_images.lua
local function build_substitution_list(image, sequence, datetime, username, pic_folder, home, desktop)
   -- build the argument substitution list from each image
   -- local datetime = os.date("*t")
   local colorlabels = {}
   if image.red then table.insert(colorlabels, "red") end
   if image.yellow then table.insert(colorlabels, "yellow") end
   if image.green then table.insert(colorlabels, "green") end
   if image.blue then table.insert(colorlabels, "blue") end
   if image.purple then table.insert(colorlabels, "purple") end
   local labels = #colorlabels == 1 and colorlabels[1] or du.join(colorlabels, ",")
   local eyear,emon,eday,ehour,emin,esec = string.match(image.exif_datetime_taken, "(%d-):(%d-):(%d-) (%d-):(%d-):(%d-)$")
   local replacements = {image.film,
                         image.path,
                         df.get_filename(image.filename),
                         string.upper(df.get_filetype(image.filename)),
                         image.id,image.duplicate_index,
                         string.format("%04d", sequence),
                         datetime.year,
                         string.format("%02d", datetime.month),
                         string.format("%02d", datetime.day),
                         string.format("%02d", datetime.hour),
                         string.format("%02d", datetime.min),
                         string.format("%02d", datetime.sec),
                         eyear,
                         emon,
                         eday,
                         ehour,
                         emin,
                         esec,
                         image.rating,
                         labels,
                         image.exif_maker,
                         image.exif_model,
                         image.title,
                         image.creator,
                         image.publisher,
                         image.rights,
                         username,
                         pic_folder,
                         home,
                         desktop,
                         image.exif_iso,
                         image.exif_exposure,
                         image.exif_exposure_bias,
                         image.exif_aperture,
                         image.exif_focus_distance,
                         image.exif_focal_length,
                         image.longitude,
                         image.latitude,
                         image.elevation,
                         image.exif_lens,
                         image.description,
                         image.exif_crop
                       }

  for i=1,#NDRL.placeholders,1 do
    NDRL.substitutes[NDRL.placeholders[i]] = replacements[i]
  end
end

local function substitute_list(str)
  -- replace the substitution variables in a string
  for match in string.gmatch(str, "%$%(.-%)") do
    local var = string.match(match, "%$%((.-)%)")
    if NDRL.substitutes[var] then
      str = string.gsub(str, "%$%("..var.."%)", NDRL.substitutes[var])
    else
      dt.print_error(_("unrecognized variable " .. var))
      dt.print(_("unknown variable " .. var .. ", aborting..."))
      return -1
    end
  end
  return str
end

local function clear_substitute_list()
  for i=1,#NDRL.placeholders,1 do NDRL.substitutes[NDRL.placeholders[i]] = nil end
end



-- perform nind-denoise and GMIC RL-decon on a single exported image ----------------------------------
local function store(storage, image, img_format, temp_name, img_num, total, hq, extra)
  if img_format.extension == "tif" and img_format.bpp ~= 16 and img_format.bpp ~= 8 then
    dt.print_log(_("ERROR: Please set TIFF bit depth to 8 or 16"))
    dt.print(_("ERROR: Please set TIFF bit depth to 8 or 16"))
    os.remove(temp_name)
    return false
  end

  local org_temp_name = temp_name
  local to_delete = {}
  table.insert(to_delete, temp_name)

  local denoise_name, tmp_rl_name, new_name, run_cmd, result
  local input_file, output_file, options

  -- determine output format
  local file_ext = img_format.extension   -- tiff only

  if extra.denoise_enabled or extra.rl_deblur_enabled then
    if extra.output_format == 1 then
      file_ext = "jpg"
    else
      file_ext = "tif"
    end
  end

  new_name = extra.output_folder..PS..df.get_basename(temp_name).."."..file_ext

  -- override output path/filename as needed
  if extra.output_path ~= "" then
    local output_path = extra.output_path
    local datetime = os.date("*t")

    build_substitution_list(image, img_num, datetime, USER, PICTURES, HOME, DESKTOP)
	  output_path = substitute_list(output_path)

	  if output_path == -1 then
	    dt.print(_("ERROR: unable to do variable substitution"))
	    return
	  end

    clear_substitute_list()
    new_name = df.get_path(output_path)..df.get_basename(output_path).."."..file_ext
  end

  dt.print_log('new_name: '..new_name)


  -- denoise
  if extra.denoise_enabled then
    if extra.nind_denoise == "" then
      dt.print(_("ERROR: nind-denoise command not configured"))
      return
    end

    -- output to TIFF if we will be debluring
    local tmp_file_ext = file_ext
    if extra.rl_deblur_enabled then
      tmp_file_ext = 'tif'
    end

    local denoise_name = df.create_unique_filename(df.get_path(temp_name)..PS..df.get_basename(temp_name).."_denoised."..tmp_file_ext)

    -- build the denoise command string
    input_file = df.sanitize_filename(temp_name)
    output_file = df.sanitize_filename(denoise_name)

    dt.print(_("denoising ")..output_file.." ...")
    run_cmd = extra.nind_denoise.." --input "..input_file.." --output "..output_file

    dt.print_log(run_cmd)

    result = dtsys.external_command(run_cmd)

    temp_name = denoise_name
    table.insert(to_delete, temp_name)

    local f = io.open(denoise_name, "r")
    if f ~= nil then
      io.close(f)
    else
      dt.print(_("Error denoising"))
      return false
    end
  end


  -- RL deblur
  if extra.rl_deblur_enabled then
    if extra.gmic == "" then
      dt.print(_("ERROR: GMic executable not configured"))
      return false
    end

    local gmic_operation = " -deblur_richardsonlucy "..extra.sigma_str..","..extra.iterations_str..",1"

    -- work around GMIC's long/space filename problem by renaming/moving file later
    local tmp_rl_name = df.create_unique_filename(df.get_path(temp_name)..PS..df.get_basename(temp_name).."_rl."..file_ext)

    -- build the GMic command string
    input_file = df.sanitize_filename(temp_name)
    output_file = df.sanitize_filename(tmp_rl_name)
    options = " cut 0,255 round "

    -- need this for 16-bit TIFF
    if extra.denoise_enabled or (img_format.extension == "tif" and img_format.bpp == 16) then
      options = " -/ 256 "..options
    end

    dt.print(_("applying RL-deblur to image ")..output_file.." ...")
    run_cmd = extra.gmic.." "..temp_name..gmic_operation..options.." -o "..output_file..","..extra.jpg_quality_str

    dt.print_log(run_cmd)

    temp_name = tmp_rl_name
    table.insert(to_delete, temp_name)

    result = dtsys.external_command(run_cmd)
    if result ~= 0 then
      dt.print(_("Error applying RL-deblur"))
      return false
    end
  end


  -- copy exif
  if extra.exiftool ~= "" then
    dt.print(_("copying EXIF to ")..temp_name.." ...")
    run_cmd = extra.exiftool.." -writeMode cg -TagsFromFile "..org_temp_name.." -all:all -overwrite_original "..df.sanitize_filename(temp_name)

    result = dtsys.external_command(run_cmd)
    if result ~= 0 then
      dt.print(_("error copying exif"))
      return false
    end
  end


  -- move the tmp file to final destination
  df.mkdir(df.sanitize_filename(df.get_path(new_name)))
  new_name = df.create_unique_filename(new_name)
  df.file_move(temp_name, new_name)
  dt.print(_("renamed and moved file to: ")..new_name)


  -- delete temp image
  for i=1,#to_delete,1 do
    os.remove(to_delete[i])
  end

  dt.print(_("finished exporting image ")..new_name)
end


-- script_manager integration

local function destroy()
  dt.destroy_storage("exp2NRL")
end






-- new widgets ----------------------------------------------------------------
local storage_widget = dt.new_widget("box"){
  orientation = "vertical",
  NDRL.output_folder_path,
  NDRL.output_folder_selector,
  NDRL.output_format,
  NDRL.jpg_quality_slider,
  NDRL.denoise_chkbox,
  NDRL.rl_deblur_chkbox,
  NDRL.sigma_slider,
  NDRL.iterations_slider
}



-- setup export ---------------------------------------------------------------
local function initialize(storage, img_format, image_table, high_quality, extra)
  local tmp_rl_name, new_name, run_cmd, result
  local input_file, output_file, options

  -- since we cannot change the bpp, inform user
  if img_format.extension == "tif" and img_format.bpp ~= 16 and img_format.bpp ~= 8 then
    dt.print_log(_("ERROR: Please set TIFF bit depth to 8 or 16"))
    dt.print(_("ERROR: Please set TIFF bit depth to 8 or 16"))
    -- not returning {} here as that can crash darktable if user clicks the export button repeatedly
  end

  -- read parameters
  extra.nind_denoise  = dt.preferences.read(MODULE_NAME, "nind_denoise", "string")
  extra.gmic          = dt.preferences.read(MODULE_NAME, "gmic_exe", "string")
  extra.gmic          = df.sanitize_filename(extra.gmic)
  extra.exiftool      = dt.preferences.read(MODULE_NAME, "exiftool_exe", "string")


  -- determine output path
  extra.output_folder = NDRL.output_folder_selector.value
  extra.output_path   = NDRL.output_folder_path.text
  extra.output_format = NDRL.output_format.selected

  extra.denoise_enabled     = NDRL.denoise_chkbox.value
  extra.rl_deblur_enabled   = NDRL.rl_deblur_chkbox.value
  extra.sigma_str           = string.gsub(string.format("%.2f", NDRL.sigma_slider.value), ",", ".")
  extra.iterations_str      = string.format("%.0f", NDRL.iterations_slider.value)
  extra.jpg_quality_str     = string.format("%.0f", NDRL.jpg_quality_slider.value)

  -- since we cannot change the bpp, inform user
  if extra.rl_deblur_enabled then
    if img_format.extension == "tif" and img_format.bpp ~= 16 and img_format.bpp ~= 8 then
      dt.print_log(_("ERROR: Please set TIFF bit depth to 8 or 16 for GMic deblur"))
      dt.print(_("ERROR: Please set TIFF bit depth to 8 or 16"))
      -- not returning {} here as that can crash darktable if user clicks the export button repeatedly
    end
  end

  -- save preferences
  dt.preferences.write(MODULE_NAME, "output_path", "string", extra.output_path)
  dt.preferences.write(MODULE_NAME, "output_format", "integer", extra.output_format)
  dt.preferences.write(MODULE_NAME, "jpg_quality", "string", extra.jpg_quality_str)
  dt.preferences.write(MODULE_NAME, "denoise_enabled", "bool", extra.denoise_enabled)
  dt.preferences.write(MODULE_NAME, "rl_deblur_enabled", "bool", extra.rl_deblur_enabled)
  dt.preferences.write(MODULE_NAME, "sigma", "string", extra.sigma_str)
  dt.preferences.write(MODULE_NAME, "iterations", "string", extra.iterations_str)

end


-- register new storage -------------------------------------------------------
dt.register_storage("exp2NDRL", _("nind-denoise RL"), store, nil, supported, initialize, storage_widget)

-- register the new preferences -----------------------------------------------
dt.preferences.register(MODULE_NAME, "nind_denoise", "string",
_ ("nind_denoise command (NRL)"),
_ ("command line to execute NIND-denoise (include --model-path"), "")

dt.preferences.register(MODULE_NAME, "gmic_exe", "file",
_ ("GMic executable (NRL)"),
_ ("select executable for GMic command line "), "")

dt.preferences.register(MODULE_NAME, "exiftool_exe", "file",
_ ("exiftool executable (NRL)"),
_ ("select executable for exiftool command line "), "")

-- set output_folder_path to the last used value at startup ------------------
NDRL.output_folder_path.text = dt.preferences.read(MODULE_NAME, "output_path", "string")

-- set output format
NDRL.output_format.selected = dt.preferences.read(MODULE_NAME, "output_format", "integer")
NDRL.jpg_quality_slider.value = dt.preferences.read(MODULE_NAME, "jpg_quality", "float")
output_format_changed()

-- set denoise and deblur options
NDRL.denoise_chkbox.value = dt.preferences.read(MODULE_NAME, "denoise_enabled", "bool")
NDRL.rl_deblur_chkbox.value = dt.preferences.read(MODULE_NAME, "rl_deblur_enabled", "bool")
denoise_rldeblur_toggled()

-- set sliders to the last used value at startup ------------------------------
NDRL.sigma_slider.value = dt.preferences.read(MODULE_NAME, "sigma", "float")
NDRL.iterations_slider.value = dt.preferences.read(MODULE_NAME, "iterations", "float")


-- script_manager integration

script_data.destroy = destroy

return script_data

-- end of script --------------------------------------------------------------

-- vim: shiftwidth=2 expandtab tabstop=2 cindent syntax=lua
-- kate: hl Lua;
