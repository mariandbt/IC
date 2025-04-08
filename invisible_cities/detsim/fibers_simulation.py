
import numpy              as np
import pandas             as pd

import math

from pint import UnitRegistry


unit = UnitRegistry()
class InvalidUnitInputError(Exception):
    pass


# *********************************************************************************
# Functions
# *********************************************************************************

def create_or_update_global_variable(global_vars, var_name, var_value, verbose):

    if verbose:
        if var_name in global_vars:
            print(f"{var_name} already exists! Updating variable.")
        else:
            print(f"{var_name} doesn't exist yet! Creating variable.")

    # Set the variable as global or update its value
    global_vars[var_name] = var_value

create_or_update_global_variable = np.vectorize(create_or_update_global_variable) # Vectorize the function



def read_fiber_sens(file_path):
    """
    Read an HDF5 file using pd.read_hdf() with an optional condition.

    Parameters:
    - file_path (str): The path to the HDF5 file.

    Returns:
    - DataFrame: The loaded DataFrame.
    """

    sns_positions   = pd.read_hdf(file_path, "/MC/sns_positions", where='sensor_name == F_SENSOR')
    sns_response    = pd.read_hdf(file_path, "/MC/sns_response")

    sns_response    = sns_response.loc[sns_response.sensor_id.isin(sns_positions.sensor_id)] # get the positions of said sensors


    return sns_positions, sns_response



def FindRotation(Electron_Final_Alpha, TPC):

    dtheta = TPC.DeltaTheta.magnitude
    
    rot = np.where(Electron_Final_Alpha > 0, 
                   int((Electron_Final_Alpha + dtheta/2)/dtheta),
                   int((Electron_Final_Alpha - dtheta/2)/dtheta)
                  )
    return rot
FindRotation = np.vectorize(FindRotation) # Vectorize the function



def BuildSensorsDict(TPC):
    
    x_dict      = dict(zip(TPC.SensorsIDs, TPC.SensorX.magnitude))
    y_dict      = dict(zip(TPC.SensorsIDs, TPC.SensorY.magnitude))

    sensor_id_sorted_by_theta   = TPC.SensorsIDs[np.argsort(TPC.SensorTheta.magnitude)]
    pos_to_id_dict              = dict(zip(TPC.SensorsPosID,  sensor_id_sorted_by_theta))
    id_to_pos_dict              = dict(zip(sensor_id_sorted_by_theta,  TPC.SensorsPosID))

    dict_names = ('x_dict', 'y_dict', 'pos_to_id_dict', 'id_to_pos_dict')
    dicts = (x_dict, y_dict, pos_to_id_dict, id_to_pos_dict)

    create_or_update_global_variable(globals(), dict_names, dicts, verbose = False)

def FindSensor(sensor_id, rotation, TPC):

    pos     = id_to_pos_dict[sensor_id]

    new_pos = pos - rotation*TPC.SensorsPerPanel

    if (new_pos == TPC.NSensors/2):
        new_pos = -TPC.NSensors/2

    if (new_pos > TPC.NSensors/2):
        new_pos = new_pos%(TPC.NSensors/2)

    if (new_pos < -TPC.NSensors/2):
        new_pos = new_pos%(-TPC.NSensors/2)

    new_sens_id = pos_to_id_dict[new_pos]

    return new_sens_id
FindSensor = np.vectorize(FindSensor) # Vectorize the function

def FindS2(sensor_id, nexusEvent, s2Table, TPC):

    rotation    = FindRotation(nexusEvent.ElectronsFinalAlpha.magnitude, TPC)

    new_alpha   = nexusEvent.ElectronsFinalAlpha - rotation * TPC.DeltaTheta
    new_xx      = nexusEvent.ElectronsFinalR * np.cos(new_alpha)
    new_yy      = nexusEvent.ElectronsFinalR * np.sin(new_alpha)

    new_sens_id     = FindSensor(sensor_id, rotation, TPC)
    
    x_bin = ((new_xx - s2Table.MinimumX)//(s2Table.WidthBinX)).magnitude.astype(int)
    y_bin = ((new_yy - s2Table.MinimumY)//(s2Table.WidthBinY)).magnitude.astype(int)
    
    zero_signal_conditions = (new_xx > s2Table.MaximumX) | \
                             (new_xx < s2Table.MinimumX) | \
                             (new_yy > s2Table.MaximumY) | \
                             (new_yy < s2Table.MinimumY)
    
    # Initialize the output array with zeros
    s2_signal = np.zeros_like(new_xx)

    # Apply the conditions
    valid_indices = ~zero_signal_conditions
    
    # For the valid indices, get the corresponding values from the s2Table.S2TablesDict
    signal_values = np.array([s2Table.S2TablesDict[f'sens_{sid}'][xb, yb] 
                              for sid, xb, yb 
                              in zip(new_sens_id[valid_indices], x_bin[valid_indices], y_bin[valid_indices])])

    # Generate Poisson-distributed random numbers for these values
    poisson_fluctuations = np.random.poisson(signal_values)

    # Place the Poisson values into the s2_signal array
    s2_signal[valid_indices] = poisson_fluctuations

    return s2_signal

def ResponseSiPM(q_in_pes, t, t0, rise_time, decay_time):

    """
    NOTE: units of t, t0, rise_time and decay_time must be the same
    """
    rise_term = 1 - np.exp(-(t - t0) / rise_time)
    decay_term = np.exp(-(t - t0) / decay_time)

    signal = (rise_term * decay_term)
    signal[t<t0] = 0
    signal_area = np.trapz(x = t, y = signal) or 1

    normalized_signal = q_in_pes*signal/signal_area

    return normalized_signal


def ConvertTomA(Waveform_in_pes_per_ns):

    sipm_gain = 4e6 # e/pes
    e_charge = 1.6e-19 # C/e
    
    pes_to_C = sipm_gain*e_charge # C/pes
    ns_to_s = 1e-9 # s/ns
    
    Waveform_in_A = Waveform_in_pes_per_ns*(pes_to_C/ns_to_s) # A
    Waveform_in_mA = Waveform_in_A*1e3 # mA

    return Waveform_in_mA
ConvertTomA = np.vectorize(ConvertTomA) # Vectorize the function


def ConvertTomV(Waveform_in_pes_per_ns, impedance_in_ohm = 50):

    Waveform_in_mV = ConvertTomA(Waveform_in_pes_per_ns)*impedance_in_ohm # mV

    return Waveform_in_mV
ConvertTomV = np.vectorize(ConvertTomV) # Vectorize the function


# *********************************************************************************
### Fiber Barrel Light Table
# *********************************************************************************
class s2LT:
    def __init__(self, s2_table_path):

        self.FilePath       = s2_table_path

        config      = pd.read_hdf(self.FilePath, "/s2LT/Config")
        s2_tab      = pd.read_hdf(self.FilePath, "/s2LT/LightTable")
        self.LT     = s2_tab
        self.Config = config

        self.LightYield     = config.loc[config.ParameterName == 'S2LightYield', 'ParameterValue'] # [ph/ie⁻]
        self.Nie            = config.loc[config.ParameterName == 'S2TableStatistics', 'ParameterValue'] # number of ie⁻ used to create the table

        self.SensorIDs  = s2_tab.filter(like="SiPMf").columns.str.replace('SiPMf', 'sens')

        self.NBinsX     = len(s2_tab.bin_initial_x.unique())
        self.NBinsY     = len(s2_tab.bin_initial_y.unique())

        self.WidthBinX  = (s2_tab.bin_final_x - s2_tab.bin_initial_x)[0] *unit.mm
        self.WidthBinY  = (s2_tab.bin_final_y - s2_tab.bin_initial_y)[0] *unit.mm
        
        self.MinimumX   = s2_tab.bin_initial_x.min() *unit.mm
        self.MinimumY   = s2_tab.bin_initial_y.min() *unit.mm

        self.MaximumX   = s2_tab.bin_final_x.max() *unit.mm
        self.MaximumY   = s2_tab.bin_final_y.max() *unit.mm

    def BuildS2TablesDict(self):

        s2_table            = self.LT
        s2_table.columns    = s2_table.columns.to_series().replace('SiPMf', 'sens', regex=True)
        s2_tab_dict = {}

        for sens_id in self.SensorIDs:

            s2_matrix = s2_table[sens_id].to_numpy().reshape(self.NBinsX, self.NBinsY)

            s2_tab_dict[sens_id] = s2_matrix
        
        self.S2TablesDict   = s2_tab_dict



# *********************************************************************************
### High Pressure Gaseus Xe TPC
# *********************************************************************************
class HPGXeTPC:
    def __init__(self, 
                 inner_rad          = 0 *unit.mm, 
                 outer_rad          = 500 *unit.mm, 
                 initial_phi        = 0*unit.rad, 
                 delta_phi          = 2*math.pi *unit.rad, 
                 length             = 3 *unit.m,
                 n_panels           = 18
                 ):

        # We unify the units for later calculations
        self.InnerRad    = inner_rad.to(unit.mm)
        self.OuterRad    = outer_rad.to(unit.mm) 
        self.InitialPhi  = initial_phi.to(unit.rad) 
        self.DeltaPhi    = delta_phi.to(unit.rad)   
        self.Length      = length.to(unit.mm)

        self.NPanels            = n_panels
        self.DeltaTheta         = 2*math.pi/self.NPanels  *unit.rad      

        self.bb0nuEnergyInXe    = (2458 *unit.keV).to(unit.eV) # ref: (https://arxiv.org/abs/1804.01780)
        self.KrDecayEnergy      = (41.5 *unit.keV).to(unit.eV) # ref: (https://arxiv.org/abs/1804.01780)

        self.HPGXeIonization    = (22 *unit.eV).to(unit.eV) # ref: (https://doi.org/10.1016/j.nima.2009.10.076)
        self.HPGXeFano          = 0.15 # ref: (https://doi.org/10.1016/j.nima.2009.10.076)

    def SetActiveDriftVelocity(self, drift_velocity = 1 *unit.mm/unit.us):
        # ref: nexus simulation
        self.ActiveDriftVelocity = drift_velocity.to(unit.mm/unit.ns)

    def SetActiveTransDiffusion(self, drift_trans_diff = 1. *unit.mm/(unit.cm**.5)):
        # ref: nexus simulation
        self.ActiveTransDiffusion = drift_trans_diff.to(unit.mm/(unit.mm**.5))

    def SetActiveLongDiffusion(self, drift_long_diff = .3 *unit.mm/(unit.cm**.5)):
        # ref: nexus simulation
        self.ActiveLongDiffusion = drift_long_diff.to(unit.mm/(unit.mm**.5))

    def SetRecombinationFactor(self, recombi_factor = 0.026):
        # ref: (https://iopscience.iop.org/article/10.1088/1748-0221/10/03/P03025/pdf)
        self.RecombinationFactor = recombi_factor

    def SetElectronLifetime(self, lifetime = 1000 *unit.ms):
        # ref: nexus default
        self.ElectronLifetime = (lifetime).to(unit.ns)

    def SetEL(self, v_drift_EL = 2.5 *unit.mm/unit.us):
        self.StartELPositionZ   = 0 *unit.mm # start of EL
        self.HalfWidthEL        = 5.1 *unit.mm
        self.DriftVelocityEL    = v_drift_EL.to(unit.mm/unit.ns)


    def SetSensors(self, sns_path):
        sns_positions, _        = read_fiber_sens(sns_path)
        self.NSensors           = len(sns_positions) # update number of sensors
        self.SensorsPerPanel    = int(self.NSensors/self.NPanels) 

        self.SensorsPosID   = np.arange((-self.NSensors/2), (self.NSensors/2))
        self.SensorsIDs     = np.array(sns_positions.sensor_id)
        self.SensorX        = np.array(sns_positions.x) *unit.mm
        self.SensorY        = np.array(sns_positions.y) *unit.mm
        self.SensorTheta    = np.round(np.arctan2(self.SensorY, self.SensorX), 3)

    def SetDefaults(self, sns_path):
        self.SetActiveDriftVelocity()
        self.SetActiveLongDiffusion()
        self.SetActiveTransDiffusion()
        self.SetRecombinationFactor()
        self.SetElectronLifetime()
        self.SetEL()
        self.SetSensors(sns_path)

        print('Default parameters set succesfully :) TPC ready to use')



# *********************************************************************************
### Nexus Event Information
# *********************************************************************************
class nexusEvent:
    def __init__(self, nexus_event_path, event_id, event_type = 'bb0nu'):

        self.EventType  = event_type
        self.EventID    = event_id

        hits            = pd.read_hdf(nexus_event_path, "/MC/hits")
        hits            = hits.query(f'event_id == {self.EventID}')

        if hits.empty:
            print('This event is empty... Try another, this one won\'t take you far :(')
            self.Empty  = True
        
        else:
            self.Empty      = False
            self.HitsX      = np.array(hits.x) *unit.mm
            self.HitsY      = np.array(hits.y) *unit.mm
            self.HitsZ      = np.array(hits.z) *unit.mm
            self.HitsTime   = np.array(hits.time) *unit.ns
            self.HitsEnergy = (np.array(hits.energy) *unit.MeV).to(unit.eV)

            self.PrimaryElectronX   = self.HitsX[self.HitsTime == self.HitsTime.min()]
            self.PrimaryElectronY   = self.HitsY[self.HitsTime == self.HitsTime.min()]
            self.PrimaryElectronR   = np.sqrt(self.PrimaryElectronX**2 + self.PrimaryElectronY**2)


    def AddDriftAndDiffusion(self, TPC):

        hit_e   = self.HitsEnergy

        t_hit   = self.HitsTime # [ns]
        x_hit   = self.HitsX # [mm]
        y_hit   = self.HitsY # [mm]
        z_hit   = self.HitsZ # [mm]
        z_EL    = TPC.StartELPositionZ # [mm] Z position towards which the e⁻ drift 

        # Drift
        z_drift = np.fabs(z_EL - z_hit) # [mm]
        v_drift = TPC.ActiveDriftVelocity
        t_drift = (z_drift/v_drift).to(unit.ns)

        t_measurement = t_hit + t_drift # After drift

        recombi_prob    = 1 - TPC.RecombinationFactor
        lifetime_prob   = np.exp(-t_drift/TPC.ElectronLifetime).magnitude

        ionization_energy   = TPC.HPGXeIonization
        fano_factor         = TPC.HPGXeFano

        n_ie    = (hit_e/ionization_energy).magnitude
        sigma   = np.sqrt(n_ie*fano_factor)

        # Fano factor correction
        n_ie    = np.random.normal(loc = n_ie*recombi_prob, scale = sigma) # recombination
        n_ie    = np.random.normal(loc = n_ie*lifetime_prob, scale = sigma) # lifetime correction

        n_ie[n_ie < 0]  = 0
        n_ie            = np.vectorize(int)(n_ie)


        self.NIonElectrons  = n_ie
        self.DriftTime      = t_drift

        # Diffusion
        drift_trans_diff = TPC.ActiveTransDiffusion
        drift_long_diff  = TPC.ActiveLongDiffusion

        sigma_trans = (drift_trans_diff*(z_drift**.5)).to(unit.mm) # [mm]
        sigma_long  = (drift_long_diff*(z_drift**.5)/v_drift).to(unit.ns) # [ns]

        self.TransDiff      = sigma_trans
        self.LongDiff       = sigma_long

        t_diff = np.array([])
        for t, std, n in zip(t_measurement, sigma_long, n_ie):
            mu      = t.magnitude
            sigma   = std.magnitude

            gaussian_diff = np.random.normal(mu, sigma, size = n)*t.units

            t_diff = np.concatenate((t_diff, gaussian_diff))
        t_diff = t_diff.to(unit.ns)

        x_diff = np.array([])
        for x, std, n in zip(x_hit, sigma_trans, n_ie):
            mu    = x.magnitude
            sigma = std.magnitude

            gaussian_diff = np.random.normal(mu, sigma, size = n)*x.units

            x_diff = np.concatenate((x_diff, gaussian_diff))
        x_diff = x_diff.to(unit.mm)

        y_diff = np.array([])
        for y, std, n in zip(y_hit, sigma_trans, n_ie):
            mu    = y.magnitude
            sigma = std.magnitude

            gaussian_diff = np.random.normal(mu, sigma, size = n)*y.units

            y_diff = np.concatenate((y_diff, gaussian_diff))
        y_diff = y_diff.to(unit.mm)

        total_n_ie  = self.NIonElectrons.sum()

        self.ElectronsIDs               = 10**(len(f'{total_n_ie}'))*self.EventID + np.arange(total_n_ie)
        self.ElectronsMeasurementTime   = t_diff.to(unit.ns)
        self.ElectronsFinalX            = x_diff.to(unit.mm)
        self.ElectronsFinalY            = y_diff.to(unit.mm)
        self.ElectronsFinalZ            = (z_EL*np.ones_like(self.ElectronsFinalY)).to(unit.mm)
        self.ElectronsFinalR            = np.sqrt(self.ElectronsFinalX**2 + self.ElectronsFinalY**2).to(unit.mm)
        self.ElectronsFinalAlpha        = np.arctan2(self.ElectronsFinalY, self.ElectronsFinalX).to(unit.rad)


# *********************************************************************************
### Signal of S2 for the Fiber Barrel sensors
# *********************************************************************************
class s2Signal:
    def __init__(self, s2LT, TPC, nexusEvent):

        s2LT.BuildS2TablesDict()
        BuildSensorsDict(TPC)
        nexusEvent.AddDriftAndDiffusion(TPC)

        self.EventType          = nexusEvent.EventType
        self.EventID            = nexusEvent.EventID
        self.SensorResponse     = {}
        self.TotalNie           = nexusEvent.NIonElectrons.sum()

        prim_e_r                = nexusEvent.PrimaryElectronR.to(unit.mm)
        self.PrimaryElectronsR  = prim_e_r.magnitude.astype(np.float32)[0] # [mm]

        if nexusEvent.NIonElectrons.sum() > 0:

            time_data   = nexusEvent.ElectronsMeasurementTime
            t_delay     = (nexusEvent.ElectronsFinalZ + TPC.HalfWidthEL)/TPC.DriftVelocityEL 
            t_delay     = t_delay.to(unit.ns)
            time_data   = (time_data + t_delay).to(unit.ns) # [ns]
            t_values    = time_data.magnitude.astype(np.float32)

            self.Time   = t_values *unit.ns# [ns]

            for jj, sens_id in enumerate(TPC.SensorsIDs[:]):

                if (((jj+1)%1 == 0) or jj == 0):
                    processing_message = f'Creating signal (Processing sensor {jj+1}/{TPC.NSensors})...'
                    print(processing_message + ' '*len(processing_message), end = '\n')

                s2_data     = FindS2(sens_id, nexusEvent, s2LT, TPC)
                s2_values   = s2_data.astype(np.float32) # [pes]

                self.SensorResponse[sens_id]    = s2_values # [pes]
            
            print(f'{processing_message} Signal created succesfully :)', end = '\n')

        else:
            print(f'This event produced no signal!')
            self.Time   = 0 *unit.ns# [ns]
            for jj, sens_id in enumerate(TPC.SensorsIDs[:]):
                self.SensorResponse[sens_id]    = 0



    def AddShapinAndSamplin(self, shapin_rise = 12 *unit.ns, shapin_tau = 28 *unit.ns, samplin_rate = 25 *unit.ns, t_binin = 0.1 *unit.ns):

        shapin_rise     = shapin_rise.to(unit.ns).magnitude
        shapin_tau      = shapin_tau.to(unit.ns).magnitude
        samplin_rate    = samplin_rate.to(unit.ns).magnitude
        t_binin         = t_binin.to(unit.ns).magnitude

        if t_binin > samplin_rate:
            raise ValueError("The time bining cannot be > the sampling rate!!")

        sensor_keys = list(self.SensorResponse.keys())
        n_sensors   = len(sensor_keys)

        time_data   = self.Time.to(unit.ns).magnitude # [ns]

        # for the s2 as deltas
        tail_in_ns  = shapin_tau*4 # [ns]
        bin_edges   = np.arange(time_data.min(), time_data.max() + tail_in_ns, t_binin)

        # for the Shaping
        bin_means               = (bin_edges[:-1] + bin_edges[1:])/2
        generic_sipm_response   = ResponseSiPM(1, bin_means, bin_means.mean(), shapin_rise, shapin_tau)

        # for the Samplin
        samplin_step = int(samplin_rate//t_binin)

        self.SignalShapedSampled    = {} 

        for jj, sensor in enumerate(sensor_keys[:]):

            if (((jj+1)%1 == 0) or jj == 0):
                processing_message = f'Shaping and sampling (Processing sensor {jj+1}/{n_sensors})...'
                print(processing_message + ' '*2*len(processing_message), end = '\n')

            s2_data     = self.SensorResponse[sensor] # [pes]

            # s2 as deltas
            s2_deltas, _    = np.histogram(time_data, bins=bin_edges, weights = s2_data)

            # Shaping
            s2_data_shaped          = np.convolve(s2_deltas, generic_sipm_response, mode='same')

            # Sample: Filter data
            s2_data_shaped_sampled  = s2_data_shaped[::samplin_step]
            s2_values               = s2_data_shaped_sampled

            self.SignalShapedSampled[sensor]    = s2_values # [pes]/[ns]
            self.SamplinRate                    = samplin_rate *unit.ns # [ns]
            self.ShapinDecayConstant            = shapin_tau *unit.ns # [ns]

        print(f'{processing_message} Shapin and samplin done succesfully :)' + ' '*10, end = '\n')

