function integrate(m1, m2, p0, e0)
    """
    General purpose integrator for eqn(5.14) of Peters, 1964 [1].

    [1] https://ui.adsabs.harvard.edu/abs/1964PhRv..136.1224P/abstract

    Params:
        m1 -- The mass of the first star  (Solar Mass)
        m2 -- The mass of the second star (Solar Mass)
        p0 -- The initial period          (days)
        e0 -- The initial eccentricity

    Returns:
        tC -- The coalescence time        (seconds)
    """
    Solar_Mass   = 1.989e30    # kg
    Solar_Radius = 696340000.0 # m
               G = 6.67e-11    # m^3 kg^-1 s^-2
               c = 299792458.0 # m/s

    convert(Float64, m1)
    convert(Float64, m2)
    convert(Float64, p0)
    convert(Float64, e0)

    ##########################################
    ############ BEGIN UNIT CHECK ############
    ##########################################
    m1 = m1 * Solar_Mass   # Kilograms     ###
    m2 = m2 * Solar_Mass   # Kilograms     ###
    p0 = p0 * 24 * 60 * 60 # Seconds       ###
    e0 = e0                # Dimensionless ###
    ##########################################
    ############# END UNIT CHECK #############
    ##########################################

    # Use Kepler's Third Law to compute the semimajor axis, in meters
    a0 = (p0^2.0 * (G * (m1+m2)) / (4.0 * pi^2.0))^(1.0/3.0) # meters

    # compute the constant c0
    c0 = a0 * (1.0-e0^2.0) * e0^(-12.0/19.0) * (1.0+(121.0 / 304.0 * e0^2.0))^(-870.0/2299.0) # meters

    # Compute the constant beta
    beta = 64.0 / 5.0 * G^3.0 * m1 * m2 * (m1 + m2) / c^5.0
    # meters^4 / s

    # Circular binary coalescence time
    tC = a0^4.0 / (4.0*beta)

    if e0 != 0
        if e0 < 0.01
            # Low eccentricity - see eqn after eqn(5.14) of Peters, 1964
            tC = c0^4.0 * e0^(48.0/19.0) / (4.0*beta)
        elseif e0 > 0.99
            # High eccentricity - see eqn after eqn after eqn(5.14) of Peters, 1964.
            tC = tC * ((768.0 / 425.0) * (1.0-e0^2.0)^3.5)
        else
            # Medium eccentricity - see eqn(5.14) of Peters, 1964
            e = 0.0
            de = e0 / 10000.0
            summand = 0.0

            while e < e0
                this_integral  = de * e^(29.0/19.0)
                this_integral *= (1.0 + (121.0/304.0) * e^2.0)^(1181.0/2299.0)
                this_integral /= (1.0 - e^2.0)^(1.5)

                summand += this_integral
                e       += de

                convert(Float64, e)
            end

            tC = (12.0/19.0) * (c0^4.0 / beta) * summand
        end
    end

    return tC
end
