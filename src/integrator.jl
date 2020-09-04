function integrate(a0, e0, p)
    """
    General purpose integrator for Nyadzani & Razzaque eqns for a & e.

    Params:
        a0              The initial semimajor axis, measured in solar radii
        e0              The initial eccentricity, dimensionless
        p               A vector of parameters:
                            p[1] = m1 (units: Solar Mass)
                            p[2] = m2 (units: Solar Mass)

    Returns:
        A               An array of the semimajor axes of the binary system over time. (Solar Mass)
        E               An array of the eccentricities of the binary system over time. (no dim.)
    """
    integrating = true

    Solar_Mass   = 1.989e30  # kg
    Solar_Radius = 696340000 # m
    G = 6.67e-11             # m^3 kg^-1 s^-2
    c = 3e8                  # m/s

    A = Float64[]
    E = Float64[]

    A = push!(A, a0 / Solar_Radius)
    E = push!(E, e0)

    m1, m2 = p[1], p[2]

    # Beta has units m^4 / s
    beta = ((64/5) * G^3 * m1 * m2 * (m1 + m2) * Solar_Mass^3 / (c^5))

    # number of seconds in a year
    secyr = 3600 * 24 * 365.25

    ## Unit Check ##
    a = a0 * Solar_Radius # Meters
    e = e0                # Dimensionless
    m1 = m1 * Solar_Mass  # Kilogram
    m2 = m2 * Solar_Mass  # Kilogram
    beta = beta           # Meters^4 / second
    ## End Unit Check ##

    total_time = 0

    while total_time/secyr < 1e11 && a > 1e4
        initial_term = (-beta / (a^3 * (1-e^2)^(7/2)))
        da = initial_term * (1 + 73/24 * e^2 + 37 / 96 * e ^ 4)

        initial_term = (-19/12 * beta / (a^4*(1-e^2)^(5/2)))
        de = initial_term * (e + 121/304 * e^3) # Units: s^-1

        timeA = abs(1e-2*a/da)
        if e > 1e-10
            timeE = abs(1e-2*e/de)
        else
            de = 0
            timeE=timeA*10e0
        end

        dt = min(timeE, timeA)

        a = a + dt * da
        e = e + dt * de

        A = push!(A, a)
        E = push!(E, e)

        total_time = total_time + dt
    end

    # Solar Radii, Dimensionless
    return A / Solar_Radius, E
end