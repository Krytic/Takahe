function Adjust(da_or_de, beta, alpha)
    return da_or_de^beta + alpha
end

str(x) = string(x)

function integrate(a0, e0, p)
    """
    General purpose integrator for Nyadzani & Razzaque eqns for a & e.

    Params:
        a0 - The initial semimajor axis, measured in solar radii
        e0 - The initial eccentricity, dimensionless
        p  - A vector of parameters:
                 p[1] = m1 (units: Solar Mass)
                 p[2] = m2 (units: Solar Mass)
                 p[3] = Beta for LT Adjustment (units: dimensionless)
                 p[4] = Alpha for LT adjustment (units: [unit]/s)
                 p[5] = Lifetime (evolution + rejuvenation)

    Returns:
        A  - An array of the semimajor axes of the binary system over time. (Solar Mass)
        E  - An array of the eccentricities of the binary system over time. (no dim.)
    """

    Solar_Mass   = 1.989e30  # kg
    Solar_Radius = 696340000.0 # m
               G = 6.67e-11  # m^3 kg^-1 s^-2
               c = 299792458.0 # m/s

               A = Float64[]
               E = Float64[]
               H = Float64[]

    m1, m2, expo, alph, evotime = p[1], p[2], p[3], p[4], p[5]

    # number of seconds in a year
    seconds_per_year = 60 * 60 * 24 * 365.25

    #######################
    ##     Unit Check     #
    #######################
    a = a0 * Solar_Radius # Meters
    e = e0                # Dimensionless
    m1 = m1 * Solar_Mass  # Kilogram
    m2 = m2 * Solar_Mass  # Kilogram
    #######################
    ##   End Unit Check   #
    #######################

    # Beta has units m^4 / s
    beta = ((64/5) * G^3 * m1 * m2 * (m1 + m2) / (c^5))

    A = push!(A, a)
    E = push!(E, e)
    H = push!(H, 0.0)

    total_time = 0

    # Integrate until past the end of the universe, or a 10km orbit
    while total_time/seconds_per_year + evotime < 1e11 && a > 1e4
        initial_da = (- beta / ( (a^3) * (1 - e^2)^(7/2) ))
        da = initial_da * (1 + (73/24) * e^2 + (37/96) * e^4)

        intial_de = (((-19/12) * beta) / (a^4*(1-e^2)^(5/2)))
        de = intial_de * (e + (121/304) * e^3) # Units: s^-1

        # da = Adjust(da, expo, alph)
        # de = Adjust(de, expo, alph)

        timeA = abs(1e-2 * a/da)
        if e > 1e-10
            timeE = abs(1e-2 * e/de)
        else
            de = 0
            e = 1e-10
            timeE = timeA * 10
        end

        dt2 = (evotime + total_time/seconds_per_year)*0.23076752*0.5*seconds_per_year

        if false
            println(string(m1 / Solar_Mass) * " " * string(m2 / Solar_Mass) * " " * string(a0 / Solar_Radius) * " " * string(e0))
            println("de=" * string(de) * ", e=" * string(e) * ", e/de=" * string(e/de))
            println("da=" * string(da) * ", a=" * string(a) * ", a/da=" * string(a/da))
            println("-------------------")
        end

        dt = min(timeE, timeA, dt2)

        a = a + dt * da
        e = e + dt * de

        A = push!(A, a)
        E = push!(E, e)
        H = push!(H, dt)

        total_time = total_time + dt
    end

    # Solar Radii, Dimensionless
    return A / Solar_Radius, E, H
end
