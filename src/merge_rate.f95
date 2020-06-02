module event_rates
      contains

      function delay_time_distribution(ensemble, edges, NBins)
            implicit none
            integer  :: NBins 
            real     :: delay_time_distribution(0:NBins)
            real     :: ensemble(0:)
            real     :: edges(0:NBins)

            integer  :: i
            real     :: merge_time
            integer  :: mergers_upto
            integer  :: culmulative_merge_rate
            integer  :: mergers_in_bin
            real     :: bin_width

            culmulative_merge_rate = 0

            ! Iterate over the bins
            do i = 0, size(ensemble)-1
                  ! Get the number of mergers up to the current bin
                  merge_time = edges(i)
                  bin_width = edges(i+1) - edges(i)

                  mergers_upto = merge_rate_f95(merge_time, ensemble)
                  mergers_in_bin = mergers_upto - culmulative_merge_rate

                  culmulative_merge_rate = culmulative_merge_rate + mergers_in_bin

                  ! normalisation
                  delay_time_distribution(i) = mergers_in_bin / 1E6 / bin_width
            enddo

            return
      end function

      function estimate_lookback(z)
            implicit none
            real    :: z, zi
            integer :: i
            real    :: integrand(0:ceiling(z/0.05))
            real    :: om, ok, ol, tH
            real    :: integral
            real    :: estimate_lookback

            om = 0.3
            ol = 0.7
            ok = 0
            tH = 1 / (70 / 3.086E19 * 315576E11)

            zi = 0
            do i=0, ceiling(z/0.5)
                  integrand(i) = 1e0 / ((1e0 + zi) * sqrt(om * (1e0+zi)**3) + ok * (1e0+zi)**2 + ol)
                  zi = zi + 0.5
            enddo

            integral = integrate(integrand, 0e0, z)
            estimate_lookback = tH * integral

            return
      end function

      function estimate_redshift(t)
            implicit none
            real    :: t
            real    :: zest
            integer :: zi
            real    :: estimate_redshift
            
            estimate_redshift = 100
            do zi = 0, 1000
                  zest = abs(estimate_lookback(float(zi) / 10) - t)
                  if(zest.le.estimate_redshift) then
                        estimate_redshift = zest
                  endif
            enddo


            return
      end function

      function compute_SFR(z1, z2)
            implicit none
            real       :: z1
            real       :: z2
            integer    :: z
            real       :: SFRD(0:ceiling(z2/0.01 - z1))
            real       :: zit
            real       :: compute_SFR
            integer    :: i

            i = 0

            do z = 100*floor(z1), 100*ceiling(z2)
                  zit = float(z) / 100
                  SFRD(i) = 0.015 * (1e0+zit)**2.7 / (1e0+((1e0+zit)/2.9)**5.6)
                  i = i + 1
            enddo

            compute_SFR = integrate(SFRD, z1, z2)

            return
      end function

      function integrate(f, x1, x2)
            implicit none
            real    :: f(0:)
            real    :: x1
            real    :: x2
            integer :: i
            real    :: integrate
            integer :: NBins
            integer :: bin_low, bin_up
            real    :: lower_bin_area, upper_bin_area
            real    :: width

            NBins = size(f)

            width = (maxval(f) - minval(f) / NBins)

            bin_low = floor(NBins * (x1 - NBins) / (NBins))
            bin_up = floor(NBins * (x2 - NBins) / (NBins))

            if (bin_low == bin_up) then
                  integrate = (x2 - x1) * f(bin_low)
                  return
            endif

            ! assume linear binning
            lower_bin_area = f(bin_low) * width
            upper_bin_area = f(bin_up) * width

            integrate = lower_bin_area + upper_bin_area

            if(bin_low + 1 /= bin_up) then
                  do i = bin_low+1, bin_up
                        integrate = integrate + f(i) * width
                  enddo
            endif

            return
      end function

      function merge_rate_f95(t_merge, ensemble)
            implicit none
            real     :: t_merge
            real     :: ensemble(0:)
            integer  :: merge_rate_f95
            integer  :: i, n

            n = size(ensemble)

            i = 0
            merge_rate_f95 = 0
            do while(i.le.n)
                  if(ensemble(i).ge.t_merge) then
                        merge_rate_f95 = merge_rate_f95 + 1
                  endif
                  i = i + 1
            enddo

            return
      end function

      subroutine event_rate_f95(events, edges, events_edges, lifetimes, lt_size, NBins)
            implicit none
            ! input parameters
            real, intent(out)   :: events(0:NBins) ! event rate distribution
            integer, intent(in) :: NBins ! number of bins
            real, intent(in)    :: events_edges(0:NBins) ! Event histogram edges
            real, intent(in)    :: edges(0:NBins) ! DTD edges
            real, intent(in)    :: lifetimes(0:lt_size-1)
            integer, intent(in) :: lt_size

            ! Custom parameters
            real                :: dtd(0:NBins)
            integer             :: i, j
            real                :: t1, z1, t1_p
            real                :: t2, z2, t2_p
            real                :: SFR
            real                :: bin_widths(0:NBins)
            integer             :: events_in_bin
            real                :: integral
            integer             :: bin

            ! functions
            dtd = delay_time_distribution(lifetimes, edges, NBins)
            print *, "DTD generated"
            do i = 1, NBins+1
                  t1 = events_edges(i-1)
                  t2 = events_edges(i)

                  print *, "Ts done"

                  z1 = estimate_redshift(t1)
                  z2 = estimate_redshift(t2)

                  print *, 'computing sfr'

                  SFR = compute_SFR(z1, z2) / (1E-3) ** 3

                  print *, "about to enter inner loop"
                  do j = 0, i-1
                        t1_p = t2 - events_edges(j)
                        t2_p = t2 - events_edges(j+1)

                        integral = integrate(dtd, t2_p, t1_p)
                        events_in_bin = floor(integral * 1E9)
                        bin = floor(NBins * (events_edges(j) - NBins) / (NBins))
                        events(bin) = events(bin) + events_in_bin + SFR
                  enddo

                  print *, "exited inner loop"
                  print *, i

                  bin_widths(i-1) = (edges(i) - edges(i-1)) * 1E9
            enddo

            do i = 0, NBins
                  events(i) = events(i) / bin_widths(i)
            enddo

      end subroutine
end module event_rates