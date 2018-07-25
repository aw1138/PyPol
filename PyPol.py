## general & for pol
import numpy as np
import glob
import pandas as pd
from astropy.io import fits
from astropy import units as u
import itertools
from astropy.time import Time

## for times
from astropy import time, coordinates as coord, units as u
from astropy.units import imperial
from astropy.units import cds

sys_err = pd.read_table('HPOL_Sys_Err_Aislynn.txt', sep=' ') 
v_fil = pd.read_table('../filters/V.fil', sep='  ', names=['wavelength', 'weight'], engine='python')

class PyPol(object):
    
    """
    A class for analyzing polarization data.
    
    Attributes:
    -----------
    
    rfiles: string
        path to the red spectra fits files
    bfiles: string
        path to the blue spectra fits files
    datekey: string
        key in the fits header for the UTC date
    mjdkey: string
        key in the fits header for the MJD date
    """
    
    def __init__(self, rfiles, bfiles, datekey='DATE-OBS', mjdkey='MJD-OBS'):
        
        self.rfiles = glob.glob(rfiles)
        self.bfiles = glob.glob(bfiles)
        self.datekey = datekey
        self.mjdkey = mjdkey
        
    def get_times(self, path):
        """
        Pulls UTC and MJD times out of a fits file. (Meant to be 
        used in conjunction with the map built-in function.)
        
        Parameters:
        -----------
        
        path: string
            path to the file
        """
        # open the file
        table = fits.open(path)

        # get the actual utc date for the observations from the header
        date = table[0].header[self.datekey]

        # get the mjd for the observations from the header
        mjd = table[0].header[self.mjdkey]

        # close the file
        table.close()

        return date, mjd
    
    def find_hjd(self):
        """
        Finds the heliocentric julian date for observations from the
        modified julian date. Parameters are containted in the parent object. 
        
        """
        # create some empty dictionaries to store information
        bobs = {}
        robs = {}

        # for every file in path given, get utc dates and mjd

        bobs['date'] = [i for i,j in list(map(self.get_times,self.bfiles))]
        bobs['mjd_blue'] = [j for i,j in list(map(self.get_times, self.bfiles))]

        robs['date'] = [i for i,j in list(map(self.get_times, self.rfiles))]
        robs['mjd_red'] = [j for i,j in list(map(self.get_times, self.rfiles))]


        bobs = pd.DataFrame(bobs) # blue observation dates, mjd
        robs = pd.DataFrame(robs) # red observation dates, mjd

        # get the mjd for dates where there is both blue and red obs

        dates = pd.merge(bobs, robs, how='inner', on=['date'])

        # get which mjd is greater for every observation
        big_blue = dates['mjd_blue'] > dates['mjd_red']
        big_red = dates['mjd_red'] > dates['mjd_blue']

        # where blue is later obs take that mjd
        mjd = dates['mjd_blue'].where(big_blue).rename(columns={'mjd_blue':'mjd'})
        utc = dates['date'].where(big_blue)

        # where red is later obs take that mjd
        temp = dates['mjd_red'].where(big_red).rename(columns={'mjd_red':'mjd'}).dropna()
        temp_u = dates['date'].where(big_red).dropna()
        #replace missing mjd slots in blue list with the ones in the red
        #this gives us the bigger mjd for every date in the table
        mjd = mjd.fillna(temp)
        utc = utc.fillna(temp_u)

        # getting mjd for dates where there is only one observation

        dates_not = pd.merge(bobs, robs, how='outer', on=['date'], 
                                left_index=True, indicator=True)
        dates_not = dates_not.query('_merge != "both"')

        # rename the columns so we can create one merged table
        mjd_not = dates_not['mjd_blue'].rename(columns={'mjd_blue':'mjd'})
        utc_not = dates_not['date']

        temp_not = dates_not['mjd_red'].rename(columns={'mjd_red':'mjd'}).dropna()
        utc_temp = dates_not['date'].dropna()

        mjd_not = mjd_not.fillna(temp_not).reset_index(drop=True) + (30 * u.min).to(cds.MJD).value
        utc_not = utc_not.fillna(utc_temp).reset_index(drop=True)

        # combine single obs date table with the double obs date table
        mjd = mjd.append(mjd_not).sort_values().reset_index(drop=True)
        utc = utc.append(utc_not).sort_values().reset_index(drop=True)

        
        # get the hjd times from the mjd times

        # specify the star location in the sky
        v367cyg = coord.SkyCoord("20:47:59.5849", "+39:17:15.723", unit=(u.hourangle, u.deg), frame='icrs')
        # specify observatory location
        loc = coord.EarthLocation.from_geodetic(43.0776*u.degree, 89.6717*u.degree, height=1188*imperial.ft)
        # turn mjd table into an astropy time object
        times = time.Time(mjd, format='mjd', scale='utc', location=loc)
        # astropy calcs the difference between hjd and mjd
        ltt_helio = times.light_travel_time(v367cyg, 'heliocentric')
        # add difference to mjd values to get hjd 
        hjd = mjd + ltt_helio.value

        # put mjd and hjd into one, shared table
        jd = {}
        jd['mjd'] = mjd
        jd['hjd'] = hjd
        jd['date'] = utc
        jd = pd.DataFrame(jd)
        
        return jd
        
    def calc_pol(self, tup, sys_err=sys_err):
        """
        A function that will calculate the angle and amount of polarized light
        given a false filter and a path to spectra data. (Meant to be 
        used in conjunction with the map built-in function.)

        Parameters:
        -----------

        tup: tuple, list of tuples
            A tuple or list of tuples of the form (filter, path). These are 
            used to obtain all relevant information that's needed.
        """
        fil, path = tup
        # tuple to make the map function work later

        # load in the table and its data
        table = fits.open(path)
        table_data = table[1].data

        # get the filter-wavelength weight
        weights = np.interp(table_data[0]['wavelength'], fil['wavelength'],
                                 fil['weight'], left=0, right=0)
        # define the flux
        flux = table[0].data

        # get the flux weight for q
        qweight = table_data[0]['q'] * flux * weights
        # sum/integrate it
        qsum = np.trapz(qweight, table_data[0]['wavelength'])

        # get the weight for just the flux
        fweight = flux * weights
        # sum it
        fsum = np.trapz(fweight, table_data[0]['wavelength'])

        # find q
        qval = qsum / fsum

        # get the flux weight for u
        uweight = table_data[0]['u'] * flux * weights
        # sum it
        usum = np.trapz(uweight, table_data[0]['wavelength'])

        # find u
        uval = usum / fsum

        # find the total polarized light
        pol_tot = np.sqrt(qval**2 + uval**2)

        # find the position angle of the light, convert to deg
        pos_ang = 0.5 * np.arctan2(uval, qval) * u.rad
        pos_ang = pos_ang.to(u.deg).value

        # grab the date to sort by later
        date = table[0].header[self.datekey]

        # define the error from the og table
        err = table_data[0]['error']

        # get weighted flux errors
        ferr = pd.DataFrame(weights * flux * err, columns=['pol_err'])
        ferr = ferr.loc[(ferr!=0).any(axis=1)]

        # get poisson error
        fish = np.sum(ferr) / np.sum(flux * weights) / np.sqrt(ferr.size)

        # turn the start and end times for sys errs into Time objs
        start = Time(sys_err['Start_Date'].values.astype(str), scale='utc')
        end = Time(sys_err['End_Date'].values.astype(str), scale='utc')

        # turn the date of obs into a time object and find which err it falls into
        time = Time(date, scale='utc')
        errmatch = (time >= start) & (time <= end)

        if all(errmatch) is False:
            t = pd.Series(start-time).append(pd.Series(end-time))
            errmatch = np.abs(t).values.argmin()
            if errmatch > 19:
                errmatch -= 19
        # get the err for the polization
        errpol = np.sqrt(fish**2 + sys_err[fil.index.name][errmatch]**2)

        # get the error for the position angle
        errang = 90 / np.pi * errpol / pol_tot

        table.close()

        return pol_tot, pos_ang, date, errpol.values[0], errang.values[0]


    def get_datepath(self, path):
        """
        Obtains the date associated with a fits file. (Meant to be 
        used in conjunction with the map built-in function.)

        Parameters:
        -----------

        path: string
            The path to the files to get dates from
        """

        table = fits.open(path)

        # grab the date of the obs and the path to the data
        tbl = {}
        tbl['date'] = [table[0].header[self.datekey]]
        tbl['path'] = [path]

        # turn them into a DataFrame
        tbl = pd.DataFrame(tbl)

        # close the file to conserve memory
        table.close()

        return tbl


    def v_band_pol(self, tup, fil=v_fil, sys_err=sys_err):
        """
        A function to do polarimetry in only the V band. (Meant to be 
        used in conjunction with the map built-in function.)

        Parameters:
        -----------

        tup: tuple of strings
            A tuple of the form (red files path, blue files path).

        fil: dictionary or DataFrame
            The filter the calculations will be done for. Default is the V filter.
        """

        # split the tuple into the paths
        path_r, path_b = tup

        # load in the red table and its data
        table_r = fits.open(path_r)
        data_r = table_r[1].data
        # grab the data needed, put into its own dataframe
        r_hold = {'wavelength': data_r[0]['wavelength'], 'q': data_r[0]['q'] , 
                  'u': data_r[0]['u'] , 'error': data_r[0]['error']}
        data_r = pd.DataFrame(r_hold)

        # load in the blue table and its data
        table_b = fits.open(path_b)
        data_b = table_b[1].data
        # grab the data needed, put into its own dataframe
        b_hold = {'wavelength': data_b[0]['wavelength'], 'q': data_b[0]['q'] , 
                  'u': data_b[0]['u'] , 'error': data_b[0]['error']}
        data_b = pd.DataFrame(b_hold)

        # get the fluxes
        flux_r = table_r[0].data
        flux_b = table_b[0].data
        # normalize one of the fluxes to match the other
        flux_b = flux_b * np.mean(flux_r) / np.mean(flux_b)

        # make the fluxes their own columns in the table
        data_r['flux'] = flux_r
        data_b['flux'] = flux_b

        # concatinate the data
        # cannot do byteswap solution to merge bc will fuck up data
        data = pd.concat([data_b, data_r], join='outer')

        # get the filter wavelength weights
        vweights = np.interp(data['wavelength'], v_fil['wavelength'], 
                                v_fil['weight'], left=0, right=0)

        # get the weight for q
        qweight = data['q'] * data['flux'] * vweights
        # sum them
        qsum = np.trapz(qweight, data['wavelength'])

        # get the flux weight
        fweight = data['flux'] * vweights
        # sum them
        fsum = np.trapz(fweight, data['wavelength'])

        # get total q
        qval = qsum / fsum

        # get u weight
        uweight = data['u'] * data['flux'] * vweights
        # sum them
        usum = np.trapz(uweight, data['wavelength'])

        # get total u
        uval = usum / fsum

        # get total polarized light
        pol_tot = np.sqrt(qval**2 + uval**2)

        # get the position angle
        pos_ang = 0.5 * np.arctan2(uval, qval) * u.rad
        pos_ang = pos_ang.to(u.deg).value

        # get the date for later
        date = table_r[0].header[self.datekey]

        # define the error from the og table
        err = data['error']
        flux = data['flux']
        # words
        s = pd.DataFrame(vweights * flux * err, columns=['pol_err'])
        s = s.loc[(s!=0).any(axis=1)]

        # get poisson error
        fish = np.sum(s) / np.sum(flux * vweights) / np.sqrt(s.size)

        # turn the start and end times for sys errs into Time objs
        start = Time(sys_err['Start_Date'].values.astype(str), scale='utc')
        end = Time(sys_err['End_Date'].values.astype(str), scale='utc')

        # turn the date of obs into a time object and find which err it falls into
        time = Time(date, scale='utc')
        errmatch = (time >= start) & (time <= end)

        if all(errmatch) is False:
            t = pd.Series(start-time).append(pd.Series(end-time))
            errmatch = np.abs(t).values.argmin()
            if errmatch > 19:
                errmatch -= 19

        # get the err for the polarization
        errpol = np.sqrt(fish**2 + sys_err['V_Band'][errmatch]**2)

        # get the error for the position angle
        errang = 90 / np.pi * errpol / pol_tot

        table_r.close()
        table_b.close()

        return pol_tot, pos_ang, date, errpol.values[0], errang.values[0]

    def find_pol(self, fils, period, sys_err, save=False, filename=None, ism=False, stars=None):
        """
        Does all the calculation for pol, pa, and their errors for all wanted filters.
        
        Parameters:
        -----------
        
        fils: list of dictionaries/pandas DataFrames
            A list containing tables/dicts with a column for filter names ('filters') and 
            another for paths ('paths'). The list is separated into red filters first, blue
            second.
        period: float
            Period of the system. Used to calculate phase.
        sys_err: string
            Path to a file containing the system error values between start and end dates.
        save: bool
            Default is False. Set to True if you want to save the table to your computer.
        filename: string
            Path + file name to save under.
        """
        sys_err = pd.read_table(sys_err, sep=' ') 
        
        ## get red filter mags
        red = fils[0]
        # create empty placeholder dictionary
        rmags = {}

        if ism is True:
            # for every red filter, get all the things from calc_pol
            for i, v in enumerate(red['paths']):
                fil = pd.read_table(v, sep='  ', names=['wavelength', 'weight'], engine='python')
                fil.index.name = red['filters'][i]+'_Band'

                # create a list of tuples with (i_fil, filename)
                prod = itertools.product([fil], self.rfiles)
                # map the function to the list of tuples
                rmags[red['filters'][i]+'pol'] = [i for i,j,k,l,m in 
                                                  [self.calc_pol_noism(x, stars) for x in prod]]

                # for some reason you have to define it again or it doesn't work
                prod = itertools.product([fil], self.rfiles)
                rmags[red['filters'][i]+'PA'] = [j for i,j,k,l,m in 
                                                 [self.calc_pol_noism(x, stars) for x in prod]]

                # create a list of tuples with (i_fil, filename)
                prod = itertools.product([fil], self.rfiles)
                # map the function to the list of tuples
                rmags[red['filters'][i]+'pol_err'] = [l for i,j,k,l,m in 
                                                      [self.calc_pol_noism(x, stars) for x in prod]]

                # for some reason you have to define it again or it doesn't work
                prod = itertools.product([fil], self.rfiles)
                rmags[red['filters'][i]+'PA_err'] = [m for i,j,k,l,m in 
                                                     [self.calc_pol_noism(x, stars) for x in prod]]

                prod = itertools.product([fil], self.rfiles)
                rmags['date'] = [k for i,j,k,l,m in [self.calc_pol_noism(x, stars) for x in prod]]

        else:
            # for every red filter, get all the things from calc_pol
            for i, v in enumerate(red['paths']):
                fil = pd.read_table(v, sep='  ', names=['wavelength', 'weight'], engine='python')
                fil.index.name = red['filters'][i]+'_Band'

                # create a list of tuples with (i_fil, filename)
                prod = itertools.product([fil], self.rfiles)
                # map the function to the list of tuples
                rmags[red['filters'][i]+'pol'] = [i for i,j,k,l,m in list(map(self.calc_pol, prod))]

                # for some reason you have to define it again or it doesn't work
                prod = itertools.product([fil], self.rfiles)
                rmags[red['filters'][i]+'PA'] = [j for i,j,k,l,m in list(map(self.calc_pol, prod))]

                # create a list of tuples with (i_fil, filename)
                prod = itertools.product([fil], self.rfiles)
                # map the function to the list of tuples
                rmags[red['filters'][i]+'pol_err'] = [l for i,j,k,l,m in list(map(self.calc_pol, prod))]

                # for some reason you have to define it again or it doesn't work
                prod = itertools.product([fil], self.rfiles)
                rmags[red['filters'][i]+'PA_err'] = [m for i,j,k,l,m in list(map(self.calc_pol, prod))]

                prod = itertools.product([fil], self.rfiles)
                rmags['date'] = [k for i,j,k,l,m in list(map(self.calc_pol, prod))]

        # convert to DataFrame
        rmags = pd.DataFrame(rmags)

        ## get blue filter mags
        blue = fils[1]
        # create empty placeholder dictionary
        bmags = {}

        if ism is True:
            # for every blue filter, get all the things from calc_pol
            for i, v in enumerate(blue['paths']):
                fil = pd.read_table(v, sep='  ', names=['wavelength', 'weight'], engine='python')
                fil.index.name = blue['filters'][i]+'_Band'

                # create a list of tuples with (i_fil, filename)
                prod = itertools.product([fil], self.bfiles)
                # map the function to the list of tuples
                bmags[blue['filters'][i]+'pol'] = [i for i,j,k,l,m in
                                                   [self.calc_pol_noism(x, stars) for x in prod]]

                # for some reason you have to define it again or it doesn't work
                prod = itertools.product([fil], self.bfiles)
                bmags[blue['filters'][i]+'PA'] = [j for i,j,k,l,m in 
                                                  [self.calc_pol_noism(x, stars) for x in prod]]

                # create a list of tuples with (i_fil, filename)
                prod = itertools.product([fil], self.bfiles)
                # map the function to the list of tuples
                bmags[blue['filters'][i]+'pol_err'] = [l for i,j,k,l,m in 
                                                       [self.calc_pol_noism(x, stars) for x in prod]]

                # for some reason you have to define it again or it doesn't work
                prod = itertools.product([fil], self.bfiles)
                bmags[blue['filters'][i]+'PA_err'] = [m for i,j,k,l,m in 
                                                      [self.calc_pol_noism(x, stars) for x in prod]]

                prod = itertools.product([fil], self.bfiles)
                bmags['date'] = [k for i,j,k,l,m in [self.calc_pol_noism(x, stars) for x in prod]]
        
        else:
            # for every blue filter, get all the things from calc_pol
            for i, v in enumerate(blue['paths']):
                fil = pd.read_table(v, sep='  ', names=['wavelength', 'weight'], engine='python')
                fil.index.name = blue['filters'][i]+'_Band'

                # create a list of tuples with (i_fil, filename)
                prod = itertools.product([fil], self.bfiles)
                # map the function to the list of tuples
                bmags[blue['filters'][i]+'pol'] = [i for i,j,k,l,m in list(map(self.calc_pol, prod))]

                # for some reason you have to define it again or it doesn't work
                prod = itertools.product([fil], self.bfiles)
                bmags[blue['filters'][i]+'PA'] = [j for i,j,k,l,m in list(map(self.calc_pol, prod))]

                # create a list of tuples with (i_fil, filename)
                prod = itertools.product([fil], self.bfiles)
                # map the function to the list of tuples
                bmags[blue['filters'][i]+'pol_err'] = [l for i,j,k,l,m in list(map(self.calc_pol, prod))]

                # for some reason you have to define it again or it doesn't work
                prod = itertools.product([fil], self.bfiles)
                bmags[blue['filters'][i]+'PA_err'] = [m for i,j,k,l,m in list(map(self.calc_pol, prod))]

                prod = itertools.product([fil], self.bfiles)
                bmags['date'] = [k for i,j,k,l,m in list(map(self.calc_pol, prod))]

        # turn the placeholder into a pandas DataFrame
        bmags = pd.DataFrame(bmags)

        ## get the v filter mags

        # get date and path for blue and red separately
        bobs = pd.concat(list(map(self.get_datepath, self.bfiles)))
        robs = pd.concat(list(map(self.get_datepath, self.rfiles)))

        # combine, take only dates where both obs happened
        v_dates = pd.merge(bobs, robs, how='inner', on=['date'], suffixes=['_b','_r'])

        # zip the paths for those dates together into a tuple
        file_tup = list(zip(v_dates['path_r'], v_dates['path_b']))

        #create a placeholder table
        vmags = {}

        # fill it with the values from the function
        vmags['Vpol'] = [i for i,j,k,l,m in list(map(self.v_band_pol, file_tup))]
        vmags['VPA'] = [j for i,j,k,l,m in list(map(self.v_band_pol, file_tup))]
        vmags['date'] = [k for i,j,k,l,m in list(map(self.v_band_pol, file_tup))]
        vmags['Vpol_err'] = [l for i,j,k,l,m in list(map(self.v_band_pol, file_tup))]
        vmags['VPA_err'] = [m for i,j,k,l,m in list(map(self.v_band_pol, file_tup))]

        # convert placeholder to a DataFrame
        vmags = pd.DataFrame(vmags)

        # merge r, b, and v tables
        final = pd.merge(rmags, bmags, how='outer', on=['date'])
        final = final.merge(vmags, how='outer', on=['date'])
        
        # merge times in
        times = self.find_hjd()
        calc = (times['hjd'] - 34266.337) / period
        phase = calc - np.trunc(calc)
        times['phase'] = phase
        
        # merge times into result table
        final = final.merge(times, on=['date'])
        # sort by increasing date and reset index
        final = final.sort_values(by=['date']).reset_index(drop=True)
        
        # save the merged table
        if save is True:
            if filename is None:
                print('ValueError: filename not defined.')
            else:
                final.to_csv(filename)

        return final
        
    def plot_lcs(self):
        
        print('placeholder')
        
    def weighted_avg(self, values, weights):
        """
        Calculates a weighted average.
        """
        avg = np.sum(values * weights) / weights.sum()
        return avg
    
    def calc_ism(self, stars_pol, stars_pa, pol_err, k, wave_max, wavelen):
        """
        Calculates the ISM average polarization and position angle.
        
        Parameters:
        -----------
        
        stars_pol: list, array
            List of polarization measurements for the chosen ism stars.
        stars_pa: list, array
            List of position angle measurements for the chosen ism stars in degrees.
        pol_err: list, array
            List of polarization errors of ism stars to calculate weighted averages.
        k: float
            Concavity of maximum polarization to polarization per wavelength function.
        wave_max: int
            Max wavelength for the polarization per wavelength function.
        wavelen: list, array
            List of wavelengths covered by the target data.
        """
        # find q and u from given pol and pa lists
        qism = stars_pol * np.cos(2 * (stars_pa * u.deg).to(u.rad).value)
        uism = stars_pol * np.sin(2 * (stars_pa * u.deg).to(u.rad).value)

        # weights for err weighted avg
        weights = 1/pol_err**2
        
        # take a weighted average
        qism_avg = self.weighted_avg(qism, weights)
        uism_avg = self.weighted_avg(uism, weights)

        # calc pol, pa from q and u
        
        # pol = sqrt(q**2 + u**2)
        ism_pmax = np.sqrt(qism_avg**2 + uism_avg**2)
        # pa = 1/2 * arctan(u/q)
        ism_pa = 1/2 * np.arctan2(uism_avg, qism_avg)
        ism_pa = (ism_pa * u.rad).to(u.deg)

        # find pol per wavelength
        ism_pol = ism_pmax * np.exp(-k * (np.log10(wave_max/wavelen)**2))
        
        # convert back to q and u
        
        # q=pol*cos(2*position angle)
        ism_q = ism_pol[0] * np.cos(2 * ism_pa.to(u.rad).value)
        # u=pol*sin(2*position angle)
        ism_u = ism_pol[0] * np.sin(2 * ism_pa.to(u.rad).value)
        
        return ism_pol, ism_pa, ism_q, ism_u
    
    def calc_pol_noism(self, target, stars, sys_err=sys_err):
        """
        A function that will calculate the angle and amount of polarized light
        given a false filter and a path to spectra data. (Meant to be 
        used in conjunction with the map built-in function.)

        Parameters:
        -----------

        target: tuple, list of tuples
            A tuple or list of tuples of the form (filter, path). These are 
            used to obtain all relevant information that's needed.
        stars: table, dictionary
            A table or dictionary of chosen ISM stars' data. Must include 
            'err', 'pol', and 'pa' columns.
        """
        fil, path = target
        # tuple to make the map function work later

        # load in the table and its data
        table = fits.open(path)
        table_data = table[1].data

        # ism calculation
        stars_pol, stars_pa, stars_err = stars
        ism_pol, ism_pa, ism_q, ism_u = self.calc_ism(stars_pol, stars_pa, stars_err, 
                                                      k=0.8486, wave_max=5100,
                                                      wavelen=table_data[0]['wavelength'])
        
        # get the filter-wavelength weight
        weights = np.interp(table_data[0]['wavelength'], fil['wavelength'],
                                 fil['weight'], left=0, right=0)
        # define the flux
        flux = table[0].data

        qog = table_data[0]['q'] - ism_pol[0]
        
        # get the flux weight for q
        qweight = qog * flux * weights
        # sum/integrate it
        qsum = np.trapz(qweight, table_data[0]['wavelength'])

        # get the weight for just the flux
        fweight = flux * weights
        # sum it
        fsum = np.trapz(fweight, table_data[0]['wavelength'])

        # find q
        qval = qsum / fsum

        uog = table_data[0]['u'] - ism_pol[0]

        # get the flux weight for u
        uweight = uog * flux * weights
        # sum it
        usum = np.trapz(uweight, table_data[0]['wavelength'])

        # find u
        uval = usum / fsum

        # find the total polarized light
        pol_tot = np.sqrt(qval**2 + uval**2)

        # find the position angle of the light, convert to deg
        pos_ang = 0.5 * np.arctan2(uval, qval) * u.rad
        pos_ang = pos_ang.to(u.deg).value

        # grab the date to sort by later
        date = table[0].header[self.datekey]

        # define the error from the og table
        err = table_data[0]['error']

        # get weighted flux errors
        ferr = pd.DataFrame(weights * flux * err, columns=['pol_err'])
        ferr = ferr.loc[(ferr!=0).any(axis=1)]

        # get poisson error
        fish = np.sum(ferr) / np.sum(flux * weights) / np.sqrt(ferr.size)

        # turn the start and end times for sys errs into Time objs
        start = Time(sys_err['Start_Date'].values.astype(str), scale='utc')
        end = Time(sys_err['End_Date'].values.astype(str), scale='utc')

        # turn the date of obs into a time object and find which err it falls into
        time = Time(date, scale='utc')
        errmatch = (time >= start) & (time <= end)

        if all(errmatch) is False:
            t = pd.Series(start-time).append(pd.Series(end-time))
            errmatch = np.abs(t).values.argmin()
            if errmatch > 19:
                errmatch -= 19
        # get the err for the polization
        errpol = np.sqrt(fish**2 + sys_err[fil.index.name][errmatch]**2)

        # get the error for the position angle
        errang = 90 / np.pi * errpol / pol_tot

        table.close()

        return pol_tot, pos_ang, date, errpol.values[0], errang.values[0]
    
    def v_pol_noism(self, tup, stars, fil=v_fil, sys_err=sys_err):
        """
        A function to do polarimetry in only the V band. (Meant to be 
        used in conjunction with the map built-in function.)

        Parameters:
        -----------

        tup: tuple of strings
            A tuple of the form (red files path, blue files path).

        fil: dictionary or DataFrame
            The filter the calculations will be done for. Default is the V filter.
        """

        # split the tuple into the paths
        path_r, path_b = tup

        # load in the red table and its data
        table_r = fits.open(path_r)
        data_r = table_r[1].data
        # grab the data needed, put into its own dataframe
        r_hold = {'wavelength': data_r[0]['wavelength'], 'q': data_r[0]['q'] , 
                  'u': data_r[0]['u'] , 'error': data_r[0]['error']}
        data_r = pd.DataFrame(r_hold)

        # load in the blue table and its data
        table_b = fits.open(path_b)
        data_b = table_b[1].data
        # grab the data needed, put into its own dataframe
        b_hold = {'wavelength': data_b[0]['wavelength'], 'q': data_b[0]['q'] , 
                  'u': data_b[0]['u'] , 'error': data_b[0]['error']}
        data_b = pd.DataFrame(b_hold)

        # get the fluxes
        flux_r = table_r[0].data
        flux_b = table_b[0].data
        # normalize one of the fluxes to match the other
        flux_b = flux_b * np.mean(flux_r) / np.mean(flux_b)

        # make the fluxes their own columns in the table
        data_r['flux'] = flux_r
        data_b['flux'] = flux_b

        # concatinate the data
        # cannot do byteswap solution to merge bc will fuck up data
        data = pd.concat([data_b, data_r], join='outer')

        # ism calculation
        stars_pol, stars_pa, stars_err = stars
        ism_pol, ism_pa, ism_q, ism_u = self.calc_ism(stars_pol, stars_pa, stars_err, 
                                                      k=0.8486, wave_max=5100,
                                                      wavelen=table_data[0]['wavelength'])
        
        # get the filter wavelength weights
        vweights = np.interp(data['wavelength'], v_fil['wavelength'], 
                                v_fil['weight'], left=0, right=0)

        # get the weight for q
        qog = data['q'] - ism_q
        qweight = qog * data['flux'] * vweights
        # sum them
        qsum = np.trapz(qweight, data['wavelength'])

        # get the flux weight
        fweight = data['flux'] * vweights
        # sum them
        fsum = np.trapz(fweight, data['wavelength'])

        # get total q
        qval = qsum / fsum

        # get u weight
        uog = data['u'] - ism_u
        uweight = uog * data['flux'] * vweights
        # sum them
        usum = np.trapz(uweight, data['wavelength'])

        # get total u
        uval = usum / fsum

        # get total polarized light
        pol_tot = np.sqrt(qval**2 + uval**2)

        # get the position angle
        pos_ang = 0.5 * np.arctan2(u, q) * u.rad
        pos_ang = pos_ang.to(u.deg).value

        # get the date for later
        date = table_r[0].header[self.datekey]

        # define the error from the og table
        err = data['error']
        flux = data['flux']
        # words
        s = pd.DataFrame(vweights * flux * err, columns=['pol_err'])
        s = s.loc[(s!=0).any(axis=1)]

        # get poisson error
        fish = np.sum(s) / np.sum(flux * vweights) / np.sqrt(s.size)

        # turn the start and end times for sys errs into Time objs
        start = Time(sys_err['Start_Date'].values.astype(str), scale='utc')
        end = Time(sys_err['End_Date'].values.astype(str), scale='utc')

        # turn the date of obs into a time object and find which err it falls into
        time = Time(date, scale='utc')
        errmatch = (time >= start) & (time <= end)

        if all(errmatch) is False:
            t = pd.Series(start-time).append(pd.Series(end-time))
            errmatch = np.abs(t).values.argmin()
            if errmatch > 19:
                errmatch -= 19

        # get the err for the polarization
        errpol = np.sqrt(fish**2 + sys_err['V_Band'][errmatch]**2)

        # get the error for the position angle
        errang = 90 / np.pi * errpol / pol_tot

        table_r.close()
        table_b.close()

        return pol_tot, pos_ang, date, errpol.values[0], errang.values[0]

