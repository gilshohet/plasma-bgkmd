module type_mod
    implicit none
    integer, parameter :: i4b = selected_int_kind(9)
    integer, parameter :: wp = kind(1.0d0)
    real(wp)::            pi = acos(-1.0_wp)
    real(wp), parameter:: el = 1.0_wp            !(au)
    real(wp), parameter:: cel = 1.0_wp/(7.29735257d-03)
    real(wp), parameter:: kBolt = 3.16499d-06        !(H.K-1)
    real(wp), parameter:: eps0 = 8.854187d-12        !(F.m-1)
    real(wp), parameter:: hbar = 1.0_wp
    ! convert to other units
    real(wp), parameter:: au_to_kg =     9.10938291d-31
    real(wp), parameter:: au_to_m =      5.2917721092d-11
    real(wp), parameter:: au_to_Ang =    5.2917721092d-01
    real(wp), parameter:: au_to_s =      2.418884326505d-17
    real(wp), parameter:: au_to_fs =     2.418884326505d-02
    real(wp), parameter:: au_to_C =      1.602176565d-19
    real(wp), parameter:: au_to_eV =     2.7211396132d+01
    real(wp), parameter:: au_to_J =      4.35974434d-18
    real(wp), parameter:: au_to_Debye =  2.541746231d+00
    real(wp), parameter:: au_to_K =      3.1577464d+05
    real(wp), parameter:: au_to_Pa =     2.9421912d+13
end module type_mod
!
!
!
!
module parameters_mod
    use type_mod ; implicit none


    ! Space-Time
    real(wp):: edgeCellSize                                     ! Size of the edge of the cell
    integer(i4b):: nxCell, nyCell, nzCell                       ! Number of cells in each direction
!    integer(i4b):: nTimeStep                                    ! Total number of timesteps
    real(wp):: timestep                                         ! Timestep


    ! Particles
    integer(i4b):: nSpecies, nParticles                         ! Number of species, Total number of particles
    real(wp):: screenLength, cutOff                             ! Yukawa screening length, cutoffLength = cutoff * screenLength
    real(wp):: extField_x, extField_y, extField_z               ! External electric field


    ! Measurements
    integer(i4b):: nOvitoTimestep                               ! Timestep jumped between two movie measurements

    !-----------------------
    !-----------------------
    ! OMP-parallelization
    integer(i4b):: nProcOMP = 0
    integer(i4b):: procNum                            ! Number of OMP processors
    real(wp), dimension(:,:,:), allocatable:: f_priv            ! private force vector : length = ( 3, nParticles, nProcOMP )
    real(wp), dimension(:,:,:,:), allocatable:: forceFSp_priv   ! private force from species : length = ( 3, nSpecies, nParticles, nProcOMP )


    ! Linked Cell List
    integer(i4b), dimension(:,:,:), allocatable:: headLCL       ! head vector for LCL : length = ( nxCell, nyCell, nzCell )
    integer(i4b), dimension(:), allocatable:: listLCL           ! list vector for LCL : length = nParticles
    integer(i4b):: nCutoffCell                                  ! Number of cells corresponding to the cutoffLength
    real(wp):: cellLength_inv                                   ! inverse of the cell length


    ! Langevin Thermostat
    logical:: thermostatOn = .false.
    
    ! MTS stuff
    real(wp):: cutOffMTS                                        ! MTS cutoff length, cutoffLength = cutoff * screenLength
    integer(i4b):: nCutoffCellMTS                               ! number of cells corresponding to the MTS cutoff length
    integer(i4b):: nTimestepMTS                                 ! number of little time steps for a big timestep
    

    ! f2py stuff
    logical:: initialized = .false.                             ! flag if system initialized and stuff allocated
    logical:: append = .false.


    ! Dummy parameters
    integer(i4b):: iTime
    real(wp):: xBoxSize, yBoxSize, zBoxSize                     ! Size of the box in each direction
    real(wp):: halfBox_x, halfBox_y, halfBox_z                  ! quantum degeneracy parameters 
    real(wp):: halfTimestep                                     ! distance to the moon (from mars)
    real(wp):: screenLength_inv                                 ! inverse of the thickness of an efficient sun screen
    logical:: outOfBounds = .false.                             ! if a particle flew out of the box

end module parameters_mod
!
!
!
!
module particles_mod
    use type_mod ; use parameters_mod ; implicit none
    
    integer(i4b), dimension(:), allocatable:: sp                 ! species index : length = nSpecies
    real(wp), dimension(:), allocatable:: mass, charge          ! mass, charge of particles : length = nParticles
    real(wp), dimension(:), allocatable:: rx, ry, rz            !
    real(wp), dimension(:), allocatable:: vx, vy, vz            !   length = nParticles
    real(wp), dimension(:), allocatable:: ax, ay, az            !
    
    real(wp), dimension(:,:), allocatable:: force               !   length = ( 3, nParticles )
    real(wp), dimension(:,:,:), allocatable:: forceFromSpecies  !   length = ( 3, nSpecies, nParticles )
    
    real(wp):: kineticE, potentialE, totalE                     ! no idea...

    ! stuff for thermostat
    real(wp), dimension(:), allocatable:: friction              ! friction term (gamma)
    real(wp), dimension(:), allocatable:: avgKE                 ! average kinetic energy for thermostat
    real(wp), dimension(:), allocatable:: ux, uy, uz            ! bulk velocity for thermostat
    real(wp), dimension(:), allocatable:: rv1, rv2, rv3, rv4    ! Random variable vectors for sampling normal distribution

    ! stuff for MTS
    integer(i4b):: headMTS                                      ! head of MTS particle list
    integer(i4b), dimension(:), allocatable:: listMTS           ! linked list of MTS particles
    logical, dimension(:), allocatable:: isMTS                  ! whether a particle s in the MTS region
    real(wp), dimension(:,:), allocatable:: forceMTS_internal   ! forces on particles due to particles inside the cutoff, length = (3, nParticles)
    real(wp), dimension(:,:), allocatable:: forceMTS_external   ! forces on particles due to particles outside the cutoff
    real(wp), dimension(:,:), allocatable:: a_tmp               ! for storing acceleration on particles within MTS area
    integer(i4b), dimension(:), allocatable:: guiltyParticles   ! closest together particles, length=2
    integer(i4b), dimension(:), allocatable:: guiltyParticles2  ! closest together particles, length=2
    integer(i4b), dimension(:,:), allocatable:: guiltyCells     ! cells corresponding to guiltyParticles, length = (2, 3)
    real(wp):: maxaij                                           ! max acceleration between two particles

end module particles_mod
!
!
!
!
subroutine evl
    use omp_lib
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    
    integer(i4b):: i
    integer(i4b), dimension(3):: iCell
    
    !$OMP PARALLEL PRIVATE(i, iCell)
    !$OMP DO
    do i= 1, nParticles
        vx(i) = vx(i) +ax(i) *halfTimestep
        rx(i) = rx(i) +vx(i) *timestep
        vy(i) = vy(i) +ay(i) *halfTimestep
        ry(i) = ry(i) +vy(i) *timestep
        vz(i) = vz(i) +az(i) *halfTimestep
        rz(i) = rz(i) +vz(i) *timestep
        
        if(abs(rx(i)) > halfBox_x) rx(i) = rx(i) -sign(xBoxSize,rx(i))
        if(abs(ry(i)) > halfBox_y) ry(i) = ry(i) -sign(yBoxSize,ry(i))
        if(abs(rz(i)) > halfBox_z) rz(i) = rz(i) -sign(zBoxSize,rz(i))

        iCell(1) = 1 +int( (rx(i) +halfBox_x)*cellLength_inv )
        iCell(2) = 1 +int( (ry(i) +halfBox_y)*cellLength_inv )
        iCell(3) = 1 +int( (rz(i) +halfBox_z)*cellLength_inv )
        if (iCell(1) < 1 .or. iCell(1) > nxCell .or. &
            iCell(2) < 1 .or. iCell(2) > nyCell .or. &
            iCell(3) < 1 .or. iCell(3) > nzCell) then
            outOfBounds = .True.
            print*, 'we lost', i, 'at position', rx(i), ry(i), rz(i)
        end if
    end do
    !$OMP END DO
    !$OMP END PARALLEL

    if (outOfBounds) return
    
    call forces
    
    !$OMP PARALLEL PRIVATE(i)
    !$OMP DO
    do i= 1, nParticles
        vx(i) = vx(i) +ax(i) *halfTimestep
        vy(i) = vy(i) +ay(i) *halfTimestep
        vz(i) = vz(i) +az(i) *halfTimestep
    end do
    !$OMP END DO
    !$OMP END PARALLEL
end subroutine evl
!
!
!
!
subroutine evlMTS
    use omp_lib
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    
    integer(i4b):: i
    integer(i4b):: parti
    integer(i4b), dimension(3):: iCell
    real(wp):: halfTimestepMTS
    real(wp):: timestepMTS
    
    halfTimestepMTS = halfTimestep / nTimestepMTS
    timestepMTS = timestep / nTimestepMTS

    ! identify particles within MTS cutoff
    call getMTSParticles

    ! evolve the MTS particle velocities based on the external forces
    call externalForcesMTS
    parti = headMTS
    do
        ax(parti) = forceMTS_external(1,parti) / mass(parti)
        ay(parti) = forceMTS_external(2,parti) / mass(parti)
        az(parti) = forceMTS_external(3,parti) / mass(parti)
        vx(parti) = vx(parti) + ax(parti) * halfTimestep
        vy(parti) = vy(parti) + ay(parti) * halfTimestep
        vz(parti) = vz(parti) + az(parti) * halfTimestep

        parti = listMTS(parti)
        if(parti==0) exit
    end do

    ! evolve the non-MTS particle velocities and positions as usual
    !$OMP PARALLEL PRIVATE(i, iCell)
    !$OMP DO
    do i= 1, nParticles
        if (isMTS(i)) cycle
        vx(i) = vx(i) +ax(i) *halfTimestep
        rx(i) = rx(i) +vx(i) *timestep
        vy(i) = vy(i) +ay(i) *halfTimestep
        ry(i) = ry(i) +vy(i) *timestep
        vz(i) = vz(i) +az(i) *halfTimestep
        rz(i) = rz(i) +vz(i) *timestep
        
        if(abs(rx(i)) > halfBox_x) rx(i) = rx(i) -sign(xBoxSize,rx(i))
        if(abs(ry(i)) > halfBox_y) ry(i) = ry(i) -sign(yBoxSize,ry(i))
        if(abs(rz(i)) > halfBox_z) rz(i) = rz(i) -sign(zBoxSize,rz(i))

        iCell(1) = 1 +int( (rx(i) +halfBox_x)*cellLength_inv )
        iCell(2) = 1 +int( (ry(i) +halfBox_y)*cellLength_inv )
        iCell(3) = 1 +int( (rz(i) +halfBox_z)*cellLength_inv )
        if (iCell(1) < 1 .or. iCell(1) > nxCell .or. &
            iCell(2) < 1 .or. iCell(2) > nyCell .or. &
            iCell(3) < 1 .or. iCell(3) > nzCell) then
            outOfBounds = .True.
            print*, 'we lost', i, 'at position', rx(i), ry(i), rz(i)
        end if
    end do
    !$OMP END DO
    !$OMP END PARALLEL
    if (outOfBounds) return

    ! take nTimestepMTS steps
    call internalForcesMTS
    do i = 1, nTimestepMTS
        parti = headMTS
        do
            ax(parti) = forceMTS_internal(1,parti) / mass(parti) + &
                extField_x*charge(parti)/mass(parti)
            ay(parti) = forceMTS_internal(2,parti) / mass(parti) + &
                extField_y*charge(parti)/mass(parti)
            az(parti) = forceMTS_internal(3,parti) / mass(parti) + &
                extField_z*charge(parti)/mass(parti)
            vx(parti) = vx(parti) + ax(parti) * halfTimestepMTS
            vy(parti) = vy(parti) + ay(parti) * halfTimestepMTS
            vz(parti) = vz(parti) + az(parti) * halfTimestepMTS
            rx(parti) = rx(parti) + vx(parti) * timestepMTS
            ry(parti) = ry(parti) + vy(parti) * timestepMTS
            rz(parti) = rz(parti) + vz(parti) * timestepMTS

            if(abs(rx(parti)) > halfBox_x) rx(parti) = rx(parti) -sign(xBoxSize,rx(parti))
            if(abs(ry(parti)) > halfBox_y) ry(parti) = ry(parti) -sign(yBoxSize,ry(parti))
            if(abs(rz(parti)) > halfBox_z) rz(parti) = rz(parti) -sign(zBoxSize,rz(parti))

            iCell(1) = 1 +int( (rx(parti) +halfBox_x)*cellLength_inv )
            iCell(2) = 1 +int( (ry(parti) +halfBox_y)*cellLength_inv )
            iCell(3) = 1 +int( (rz(parti) +halfBox_z)*cellLength_inv )
            if (iCell(1) < 1 .or. iCell(1) > nxCell .or. &
                iCell(2) < 1 .or. iCell(2) > nyCell .or. &
                iCell(3) < 1 .or. iCell(3) > nzCell) then
                print*, 'we lost', parti, 'at position', rx(parti), ry(parti), rz(parti)
                outOfBounds = .True.
                return
            end if
            parti = listMTS(parti)
            if(parti==0) exit
        end do

        call internalForcesMTS
        parti = headMTS
        do
            ax(parti) = forceMTS_internal(1,parti) / mass(parti) + &
                extField_x*charge(parti)/mass(parti)
            ay(parti) = forceMTS_internal(2,parti) / mass(parti) + &
                extField_y*charge(parti)/mass(parti)
            az(parti) = forceMTS_internal(3,parti) / mass(parti) + &
                extField_z*charge(parti)/mass(parti)
            vx(parti) = vx(parti) + ax(parti) * halfTimestepMTS
            vy(parti) = vy(parti) + ay(parti) * halfTimestepMTS
            vz(parti) = vz(parti) + az(parti) * halfTimestepMTS

            parti = listMTS(parti)
            if(parti==0) exit
        end do
    end do
    
    ! evolve the MTS particle velocities again based on the new external forces
    call externalForcesMTS
    parti = headMTS
    do
        ax(parti) = forceMTS_external(1,parti) / mass(parti)
        ay(parti) = forceMTS_external(2,parti) / mass(parti)
        az(parti) = forceMTS_external(3,parti) / mass(parti)
        vx(parti) = vx(parti) + ax(parti) * halfTimestep
        vy(parti) = vy(parti) + ay(parti) * halfTimestep
        vz(parti) = vz(parti) + az(parti) * halfTimestep

        parti = listMTS(parti)
        if(parti==0) exit
    end do
    
    ! get the forces at the end of the time step
    call forces
    
    ! update the non-MTS particle velocities the second time (MTS already done)
    !$OMP PARALLEL PRIVATE(i)
    !$OMP DO
    do i= 1, nParticles
        if (isMTS(i)) cycle
        vx(i) = vx(i) +ax(i) *halfTimestep
        vy(i) = vy(i) +ay(i) *halfTimestep
        vz(i) = vz(i) +az(i) *halfTimestep
    end do
    !$OMP END DO
    !$OMP END PARALLEL

end subroutine evlMTS
!
!
!
!
subroutine naughtyPair

    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    real(wp), dimension(3):: rij
    real(wp):: dij, mindij, dij_inv, q2ij, uij, normFij, fij, aij1, aij2
    integer(i4b):: i
    integer(i4b):: ix, iy, iz
    integer(i4b), dimension(2):: xlim, ylim
    integer(i4b):: xiCell,yiCell,ziCell, xjCell,yjCell,zjCell
    integer(i4b):: parti, partj

    guiltyParticles = 0
    guiltyParticles2 = 0
    guiltyCells = 0
    mindij = sqrt(xBoxSize**2 + yBoxSize**2 + zBoxSize**2)
    maxaij = 0.0_wp

    ! loop over the cells in x, y, z
    do ziCell = 1, nzCell
    do yiCell = 1, nyCell
    do xiCell = 1, nxCell
        parti = headLCL(xiCell,yiCell,ziCell)
        if(parti == 0) cycle
        ! loop over the particles in the cell
        do
            ! loop adjacent cells such that we only visit each pair once
            do iz = 0, 1
                zjCell = ziCell +iz
                if(zjCell > nzCell) zjCell = zjCell -nzCell
                if(zjCell < 1) zjCell = zjCell +nzCell
                if (iz == 0) then
                    ylim(1) = 0
                    ylim(2) = 1
                else
                    ylim(1) = -1
                    ylim(2) = 1
                end if
            do iy = ylim(1), ylim(2)
                yjCell = yiCell +iy
                if(yjCell > nyCell) yjCell = yjCell -nyCell
                if(yjCell < 1) yjCell = yjCell +nyCell
                if (iy == 0 .and. iz==0) then
                    xlim(1) = 0
                    xlim(2) = 1
                else
                    xlim(1) = -1
                    xlim(2) = 1
                endif
            do ix = xlim(1), xlim(2)
                xjCell = xiCell +ix
                if(xjCell > nxCell) xjCell = xjCell -nxCell
                if(xjCell < 1) xjCell = xjCell +nxCell

                partj = headLCL(xjCell,yjCell,zjCell)
                if(iz == 0 .and. iy == 0 .and. ix == 0) partj = listLCL(parti)
                if(partj == 0) cycle
                
                ! loop over particles in the other cell
                do
                    ! compute distance
                    rij(1) = rx(parti) -rx(partj)
                    rij(2) = ry(parti) -ry(partj)
                    rij(3) = rz(parti) -rz(partj)
                    if(abs(rij(1)) > halfBox_x) rij(1) = rij(1) -sign(xBoxSize, rx(parti))
                    if(abs(rij(2)) > halfBox_y) rij(2) = rij(2) -sign(yBoxSize, ry(parti))
                    if(abs(rij(3)) > halfBox_z) rij(3) = rij(3) -sign(zBoxSize, rz(parti))
                    dij = sqrt(sum(rij**2))
                    dij_inv = 1.0_wp/dij
                    q2ij = charge(parti)*charge(partj)
                    
                    uij = q2ij*exp(-dij*screenLength_inv)*dij_inv
                    normFij = uij*(1.0_wp+dij*screenLength_inv)*dij_inv
                    fij = sqrt(sum((rij*normFij)**2))
                    aij1 = fij / mass(parti)
                    aij2 = fij / mass(partj)

                    ! check if biggest force
                    if (aij1 > maxaij .or. aij2 > maxaij) then
                        maxaij = max(aij1, aij2)
                        guiltyParticles(1) = parti
                        guiltyParticles(2) = partj
                        guiltyCells(1,1) = xiCell
                        guiltyCells(1,2) = yiCell
                        guiltyCells(1,3) = ziCell
                        guiltyCells(2,1) = xjCell
                        guiltyCells(2,2) = yjCell
                        guiltyCells(2,3) = zjCell
                    end if

                    ! check if minimum distance
                    if(dij < mindij) then
                        mindij = dij
                        guiltyParticles2(1) = parti
                        guiltyParticles2(2) = partj
                    end if

                    partj = listLCL(partj)
                    if(partj == 0) exit
                end do ! loop over particles in other cell

            end do ! ix
            end do ! iy
            end do ! iz

            parti = listLCL(parti)
            if(parti == 0) exit
        end do ! loop over particles in the cell
    end do ! xiCell
    end do ! yiCell
    end do ! ziCell
    write(23,*) maxaij, sp(guiltyParticles(1)), sp(guiltyParticles(2)), guiltyParticles(1), guiltyParticles(2)
    write(21,*) mindij, sp(guiltyParticles2(1)), sp(guiltyParticles2(2)), guiltyParticles2(1), guiltyParticles2(2)

end subroutine naughtyPair
!
!
!
!
subroutine getMTSParticles

    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    integer(i4b):: xiCell,yiCell,ziCell
    integer(i4b):: ix, iy, iz, ic
    integer(i4b):: parti

    headMTS = 0
    listMTS = 0
    isMTS = .false.

    ! loop over cells within the MTS cutoff region
    do ic = 1,2
    do iz = guiltyCells(ic,3) - nCutoffCellMTS, guiltyCells(ic,3) + nCutoffCellMTS
        ziCell = iz
        if(iz > nzCell) ziCell = iz - nzCell
        if(iz < 1) ziCell = iz + nzCell
    do iy = guiltyCells(ic,2) - nCutoffCellMTS, guiltyCells(ic,2) + nCutoffCellMTS
        yiCell = iy
        if(iy > nyCell) yiCell = iy - nyCell
        if(iy < 1) yiCell = iy + nyCell
    do ix = guiltyCells(ic,1) - nCutoffCellMTS, guiltyCells(ic,1) + nCutoffCellMTS
        xiCell = ix
        if(ix > nxCell) xiCell = ix - nxCell
        if(ix < 1) xiCell = ix + nxCell

        ! loop over particles in the cell
        parti = headLCL(xiCell,yiCell,ziCell)
        if(parti == 0) cycle
        if(isMTS(parti)) cycle
        do
            ! update MTS list
            listMTS(parti) = headMTS
            headMTS = parti
            isMTS(parti) = .True.

            parti = listLCL(parti)
            if(parti == 0) exit
        end do ! loop over particles in the cell
    end do ! ix
    end do ! iy
    end do ! iz
    end do ! ic

    ! check for cyclic list
    parti = headMTS
    do
        parti = listMTS(parti)
        if (parti == 0) exit
        if (parti == headMTS) then
            print*, 'list is cyclic!'
            print*, guiltyCells(1,:)
            print*, guiltyCells(2,:)
            STOP 666
        end if
    end do

end subroutine getMTSParticles
!
!
!
!
subroutine externalForcesMTS

    use omp_lib
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    integer(i4b):: i
    integer(i4b), dimension(3):: iCell
    integer(i4b):: ix, iy, iz 
    integer(i4b):: xjCell,yjCell,zjCell
    integer(i4b):: parti, partj
    real(wp):: dij, dij_inv
    real(wp):: q2ij, uij, normFij
    real(wp), dimension(3):: rij, fij

    forceMTS_external = 0.0_wp

    parti = headMTS
    ! loop over the MTS particles
    do
        ! get which cell the particle is in
        iCell(1) = 1 +int( (rx(parti) +halfBox_x)*cellLength_inv )
        iCell(2) = 1 +int( (ry(parti) +halfBox_y)*cellLength_inv )
        iCell(3) = 1 +int( (rz(parti) +halfBox_z)*cellLength_inv )

        ! loop over cells within cutoff radius
        !$OMP PARALLEL &
        !$OMP PRIVATE(procNum, iz, iy, ix, zjCell, yjCell, xjCell, partj, rij, dij, dij_inv, q2ij, normFij, fij)
        procNum = OMP_GET_THREAD_NUM() + 1
        f_priv(:,parti,:) = 0.0_wp
        !$OMP DO
        do iz = -nCutoffCell, nCutoffCell
            zjCell = iCell(3) +iz
            if(zjCell > nzCell) zjCell = zjCell -nzCell
            if(zjCell < 1) zjCell = zjCell +nzCell
        do iy = -nCutoffCell, nCutoffCell
            yjCell = iCell(2) +iy
            if(yjCell > nyCell) yjCell = yjCell -nyCell
            if(yjCell < 1) yjCell = yjCell +nyCell
        do ix = -nCutoffCell, nCutoffCell
            xjCell = iCell(1) +ix
            if(xjCell > nxCell) xjCell = xjCell -nxCell
            if(xjCell < 1) xjCell = xjCell +nxCell

            partj = headLCL(xjCell,yjCell,zjCell)
            if(iz == 0 .and. iy == 0 .and. ix == 0) partj = listLCL(parti)
            if(partj == 0) cycle

            ! loop over other particles and compute forces
            do
                if (.not. isMTS(partj)) then
                    normFij = 0.0_wp
                    rij(1) = rx(parti) -rx(partj)
                    rij(2) = ry(parti) -ry(partj)
                    rij(3) = rz(parti) -rz(partj)
                    if(abs(rij(1)) > halfBox_x) rij(1) = rij(1) -sign(xBoxSize, rx(parti))
                    if(abs(rij(2)) > halfBox_y) rij(2) = rij(2) -sign(yBoxSize, ry(parti))
                    if(abs(rij(3)) > halfBox_z) rij(3) = rij(3) -sign(zBoxSize, rz(parti))
                    
                    dij = sqrt(sum(rij**2))
                    dij_inv = 1.0_wp/dij
                    q2ij = charge(parti)*charge(partj)
                    
                    uij = q2ij*exp(-dij*screenLength_inv)*dij_inv
                    normFij = uij*(1.0_wp+dij*screenLength_inv)*dij_inv**2
                    
                    fij = rij*normFij
                    f_priv(:,parti,procNum) = f_priv(:,parti,procNum) + fij
                end if

                partj = listLCL(partj)
                if(partj == 0) exit
            end do
        end do !iy
        end do !ix
        end do !iz
        !$OMP END DO
        !$OMP END PARALLEL

        do i = 1, nProcOMP
            forceMTS_external(:,parti) = forceMTS_external(:,parti) + f_priv(:,parti,i)
        end do

        parti = listMTS(parti)
        if(parti == 0) exit
    end do ! loop over the MTS particles

end subroutine externalForcesMTS
!
!
!
!
subroutine internalForcesMTS
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    integer(i4b):: parti, partj
    real(wp):: dij, dij_inv
    real(wp):: q2ij, uij, normFij
    real(wp), dimension(3):: rij, fij

    forceMTS_internal = 0.0_wp
    ! loop over the MTS particles
    parti = headMTS
    do
        ! loop over the subsequent MTS particles and compute forces
        partj = listMTS(parti)
        if(partj == 0) exit
        do
            normFij = 0.0_wp
            rij(1) = rx(parti) -rx(partj)
            rij(2) = ry(parti) -ry(partj)
            rij(3) = rz(parti) -rz(partj)
            if(abs(rij(1)) > halfBox_x) rij(1) = rij(1) -sign(xBoxSize, rx(parti))
            if(abs(rij(2)) > halfBox_y) rij(2) = rij(2) -sign(yBoxSize, ry(parti))
            if(abs(rij(3)) > halfBox_z) rij(3) = rij(3) -sign(zBoxSize, rz(parti))
            
            dij = sqrt(sum(rij**2))
            dij_inv = 1.0_wp/dij
            q2ij = charge(parti)*charge(partj)
            
            uij = q2ij*exp(-dij*screenLength_inv)*dij_inv
            normFij = uij*(1.0_wp+dij*screenLength_inv)*dij_inv**2
            
            fij = rij*normFij
            forceMTS_internal(:,parti) = forceMTS_internal(:,parti) + fij
            forceMTS_internal(:,partj) = forceMTS_internal(:,partj) - fij

            partj = listMTS(partj)
            if(partj == 0) exit
        end do ! loop over subsequent MTS particles

        parti = listMTS(parti)
        if(parti == 0) exit
    end do ! loop over the MTS particles

end subroutine internalForcesMTS
!
!
!
!
subroutine forces
    use omp_lib
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    
    real(wp):: dij, dij_inv
    real(wp):: q2ij, uij, normFij
    real(wp), dimension(3):: rij, fij
    integer(i4b), dimension(3):: iCell
    integer(i4b):: i
    integer(i4b):: ix,iy,iz
    integer(i4b):: xiCell,yiCell,ziCell, xjCell,yjCell,zjCell
    integer(i4b):: parti, partj

    !-------------------------------------------
    !--- LINKED CELL LIST ----------------------
    !-------------------------------------------
    headLCL=0
    listLCL=0
    do i=1, nParticles
        iCell(1) = 1 +int( (rx(i) +halfBox_x)*cellLength_inv )
        iCell(2) = 1 +int( (ry(i) +halfBox_y)*cellLength_inv )
        iCell(3) = 1 +int( (rz(i) +halfBox_z)*cellLength_inv )
        listLCL(i) = headLCL(iCell(1),iCell(2),iCell(3))
        headLCL(iCell(1),iCell(2),iCell(3)) = i
    end do
    !-------------------------------------------

    !-------------------------------------------
    !--- YUKAWA --------------------------------
    !-------------------------------------------
    potentialE=0.0_wp
    f_priv=0.0_wp
    forceFSp_priv=0.0_wp
    
    !$OMP PARALLEL &
    !$OMP PRIVATE(procNum, ziCell,xiCell,yiCell,zjCell,xjCell,yjCell, ix,iy,iz, parti,partj, &
    !$OMP rij,dij,dij_inv,q2ij,uij,normFij,fij) &
    !$OMP REDUCTION(+: potentialE)
    procNum = OMP_GET_THREAD_NUM()
    procNum = procNum +1
    !$OMP DO
    do ziCell = 1, nzCell
    do xiCell = 1, nxCell
    do yiCell = 1, nyCell
        parti = headLCL(xiCell,yiCell,ziCell)
        if(parti == 0) cycle


        do
        ! Same cell
        partj = parti
        do
            partj = listLCL(partj)
            if(partj == 0) exit
            
            normFij = 0.0_wp
            rij(1) = rx(parti) -rx(partj)
            rij(2) = ry(parti) -ry(partj)
            rij(3) = rz(parti) -rz(partj)
            dij = sqrt(sum(rij**2))
            dij_inv = 1.0_wp/dij
            q2ij = charge(parti)*charge(partj)

            uij = q2ij*exp(-dij*screenLength_inv)*dij_inv
            potentialE = potentialE +uij
            normFij = uij*(1.0_wp+dij*screenLength_inv)*dij_inv**2

            fij = rij*normFij
            f_priv(:,parti,procNum) = f_priv(:,parti,procNum) +fij
            f_priv(:,partj,procNum) = f_priv(:,partj,procNum) -fij
            
            forceFSp_priv(:,Sp(partj),parti,procNum) = forceFSp_priv(:,Sp(partj),parti,procNum) +fij
            forceFSp_priv(:,Sp(parti),partj,procNum) = forceFSp_priv(:,Sp(parti),partj,procNum) -fij            
        end do
         
        
        
        ! Different cells
        do ix = 1, nCutoffCell
            xjCell = xiCell +ix
            if(xjCell > nxCell) xjCell = xjCell -nxCell
            partj = headLCL(xjCell,yiCell,ziCell)
            if(partj == 0) cycle
            
            do
            normFij = 0.0_wp
            rij(1) = rx(parti) -rx(partj)
            rij(2) = ry(parti) -ry(partj)
            rij(3) = rz(parti) -rz(partj)
            if(abs(rij(1)) > halfBox_x) rij(1) = rij(1) -sign(xBoxSize, rx(parti))
            
            dij = sqrt(sum(rij**2))
            dij_inv = 1.0_wp/dij
            q2ij = charge(parti)*charge(partj)
            
            uij = q2ij*exp(-dij*screenLength_inv)*dij_inv
            potentialE = potentialE +uij
            normFij = uij*(1.0_wp+dij*screenLength_inv)*dij_inv**2
            
            fij = rij*normFij
            f_priv(:,parti,procNum) = f_priv(:,parti,procNum) +fij
            f_priv(:,partj,procNum) = f_priv(:,partj,procNum) -fij
            
            forceFSp_priv(:,Sp(partj),parti,procNum) = forceFSp_priv(:,Sp(partj),parti,procNum) +fij
            forceFSp_priv(:,Sp(parti),partj,procNum) = forceFSp_priv(:,Sp(parti),partj,procNum) -fij            
            
            partj = listLCL(partj)
            if(partj == 0) exit
            end do
        end do
        
        
        do iy = 1, nCutoffCell
            yjCell = yiCell +iy
            if(yjCell > nyCell) yjCell = yjCell -nyCell
        do ix = -nCutoffCell, nCutoffCell
            xjCell = xiCell +ix
            if(xjCell > nxCell) then
                xjCell = xjCell -nxCell
            else if(xjCell < 1) then
                xjCell = xjCell +nxCell
            end if
            partj = headLCL(xjCell,yjCell,ziCell)
            if(partj == 0) cycle
            
            do
            normFij = 0.0_wp
            rij(1) = rx(parti) -rx(partj)
            rij(2) = ry(parti) -ry(partj)
            rij(3) = rz(parti) -rz(partj)
            if(abs(rij(1)) > halfBox_x) rij(1) = rij(1) -sign(xBoxSize, rx(parti))
            if(abs(rij(2)) > halfBox_y) rij(2) = rij(2) -sign(yBoxSize, ry(parti))
            
            dij = sqrt(sum(rij**2))
            dij_inv = 1.0_wp/dij
            q2ij = charge(parti)*charge(partj)
            
            
            uij = q2ij*exp(-dij*screenLength_inv)*dij_inv
            potentialE = potentialE +uij
            normFij = uij*(1.0_wp+dij*screenLength_inv)*dij_inv**2
            
            
            fij = rij*normFij
            f_priv(:,parti,procNum) = f_priv(:,parti,procNum) +fij
            f_priv(:,partj,procNum) = f_priv(:,partj,procNum) -fij
            
            forceFSp_priv(:,Sp(partj),parti,procNum) = forceFSp_priv(:,Sp(partj),parti,procNum) +fij
            forceFSp_priv(:,Sp(parti),partj,procNum) = forceFSp_priv(:,Sp(parti),partj,procNum) -fij                        
            
            partj = listLCL(partj)
            if(partj == 0) exit
            end do
        end do
        end do
        
        
        do iz = 1, nCutoffCell
            zjCell = ziCell +iz
            if(zjCell > nzCell) zjCell = zjCell -nzCell
        do iy = -nCutoffCell, nCutoffCell
            yjCell = yiCell +iy
            if(yjCell > nyCell) then
                yjCell = yjCell -nyCell
            else if(yjCell < 1) then
                yjCell = yjCell +nyCell
            end if
        do ix = -nCutoffCell, nCutoffCell
            xjCell = xiCell +ix
            if(xjCell > nxCell) then
                xjCell = xjCell -nxCell
            else if(xjCell < 1) then
                xjCell = xjCell +nxCell
            end if
            partj = headLCL(xjCell,yjCell,zjCell)
            if(partj == 0) cycle
            
            
            do
            normFij = 0.0_wp
            rij(1) = rx(parti) -rx(partj)
            rij(2) = ry(parti) -ry(partj)
            rij(3) = rz(parti) -rz(partj)
            if(abs(rij(1)) > halfBox_x) rij(1) = rij(1) -sign(xBoxSize, rx(parti))
            if(abs(rij(2)) > halfBox_y) rij(2) = rij(2) -sign(yBoxSize, ry(parti))
            if(abs(rij(3)) > halfBox_z) rij(3) = rij(3) -sign(zBoxSize, rz(parti))
            
            dij = sqrt(sum(rij**2))
            dij_inv = 1.0_wp/dij
            q2ij = charge(parti)*charge(partj)
            
            uij = q2ij*exp(-dij*screenLength_inv)*dij_inv
            potentialE = potentialE +uij
            normFij = uij*(1.0_wp+dij*screenLength_inv)*dij_inv**2
            
            fij = rij*normFij
            f_priv(:,parti,procNum) = f_priv(:,parti,procNum) +fij
            f_priv(:,partj,procNum) = f_priv(:,partj,procNum) -fij
            
            forceFSp_priv(:,Sp(partj),parti,procNum) = forceFSp_priv(:,Sp(partj),parti,procNum) +fij
            forceFSp_priv(:,Sp(parti),partj,procNum) = forceFSp_priv(:,Sp(parti),partj,procNum) -fij            

            partj = listLCL(partj)
            if(partj == 0) exit
            end do
        end do !iy
        end do !ix
        end do !iz
        
        
        
        parti = listLCL(parti)
        if(parti == 0) exit
        end do
    end do ! xiCell = 1, nbCell_x
    end do ! yiCell = 1, nbCell_y
    end do ! ziCell = 1, nbCell_z
    !$OMP END DO
    !$OMP END PARALLEL
    !-------------------------------------------
    
    force=0.0_wp
    forceFromSpecies=0.0_wp
    do i = 1, nProcOMP
       force(:,:) = force(:,:) +f_priv(:,:,i)
       forceFromSpecies(:,:,:) = forceFromSpecies(:,:,:) +forceFSp_priv(:,:,:,i)
    end do
    ax = force(1,:)/mass + extField_x*charge/mass
    ay = force(2,:)/mass + extField_y*charge/mass
    az = force(3,:)/mass + extField_z*charge/mass

    ! langevin stuff
    if (thermostatOn) then
        call random_number(rv1) ; call random_number(rv2) 
        call random_number(rv3) ; call random_number(rv4)
        ax = ax - friction * (vx - ux) &
            + sqrt(4.0_wp * friction * avgKE / (3.0_wp * mass * timestep)) &
            * sqrt(-2.0_wp * log(rv1)) * cos(2.0_wp * pi * rv2)
        ay = ay - friction * (vy - uy) &
            + sqrt(4.0_wp * friction * avgKE / (3.0_wp * mass * timestep)) &
            * sqrt(-2.0_wp * log(rv1)) * sin(2.0_wp * pi * rv2)
        az = az - friction * (vz - uz) &
            + sqrt(4.0_wp * friction * avgKE / (3.0_wp * mass * timestep)) &
            * sqrt(-2.0_wp * log(rv3)) * cos(2.0_wp * pi * rv4)
    end if

end subroutine forces
!
!
!
!
subroutine initOMP
    use omp_lib
    use type_mod ; use particles_mod ; implicit none
    
    integer(i4b):: i

    ! OMP
    nProcOMP = min(nProcOMP, OMP_GET_NUM_PROCS())                         !<------ Creates as many threads as there are free cores in the system
    print*, '- Nb of available procs = ', nProcOMP
    
    call OMP_SET_NUM_THREADS(nProcOMP)
    
    !$OMP PARALLEL
    print*, 'I am thread', OMP_GET_THREAD_NUM(), 'In parallel ?', OMP_IN_PARALLEL()
    !$OMP END PARALLEL

end subroutine initOMP
!
!
!
!
subroutine iniSimulation
    use type_mod ; use particles_mod ; implicit none

    integer(i4b):: i

    initialized = .true.

    ! Space-Time
    cellLength_inv = 1.0_wp/edgeCellSize
    xBoxSize = real(nxCell)*edgeCellSize ; halfBox_x = 0.5_wp*xBoxSize
    yBoxSize = real(nyCell)*edgeCellSize ; halfBox_y = 0.5_wp*yBoxSize
    zBoxSize = real(nzCell)*edgeCellSize ; halfBox_z = 0.5_wp*zBoxSize
    allocate(headLCL(nxCell,nyCell,nzCell)) ; headLCL=0

    halfTimestep = 0.5_wp*timestep

    allocate(listLCL(nParticles)) ; listLCL=0
    allocate(sp(nParticles), mass(nParticles), charge(nParticles)) ; sp=0 ; mass=0.0_wp ; charge=0.0_wp
    allocate(rx(nParticles), ry(nParticles), rz(nParticles)) ; rx=0.0_wp ; ry=0.0_wp ; rz=0.0_wp
    allocate(vx(nParticles), vy(nParticles), vz(nParticles)) ; vx=0.0_wp ; vy=0.0_wp ; vz=0.0_wp
    allocate(ax(nParticles), ay(nParticles), az(nParticles)) ; ax=0.0_wp ; ay=0.0_wp ; az=0.0_wp
    allocate(force(3, nParticles)) ; force=0.0_wp
    allocate(forceFromSpecies(3, nSpecies, nParticles)) ; forceFromSpecies=0.0_wp
    allocate(f_priv(3, nParticles, nProcOMP)) ; f_priv=0.0_wp
    allocate(forceFSp_priv(3, nSpecies, nParticles, nProcOMP)) ; forceFSp_priv=0.0_wp

    ! for MTS
    allocate(listMTS(nParticles)) ; listMTS = 0
    allocate(isMTS(nParticles)) ; isMTS = .false.
    allocate(forceMTS_internal(3, nParticles)) ; forceMTS_internal = 0.0_wp
    allocate(forceMTS_external(3, nParticles)) ; forceMTS_external = 0.0_wp
    allocate(a_tmp(3, nParticles)) ; a_tmp = 0.0_wp
    allocate(guiltyParticles(2)) ; guiltyParticles = 0.0_wp
    allocate(guiltyParticles2(2)) ; guiltyParticles2 = 0.0_wp
    allocate(guiltyCells(2, 3)); guiltyCells = 0.0_wp

    screenLength_inv = 1.0_wp/screenLength
    nCutoffCell = int(cutoff*screenlength/edgeCellSize) +1
    if(nCutoffCell > min(nxCell,nyCell,nzCell)/2) then
        write(*,*) 'nCutoffCell > nCellxyz/2'
        STOP 11111
    end if

    nCutoffCellMTS = int(cutoffMTS*screenlength/edgecellsize) + 1
    if (cutoffMTS == 0) nCutoffCellMTS = 0

    !----------------------------------------
    ! fill Langevin thermostat variables if needed
    if (thermostatOn) then
        allocate(friction(nParticles)) ; friction=0.0_wp
        allocate(avgKE(nParticles)) ; avgKE = 0.0_wp
        allocate(ux(nParticles), uy(nParticles), uz(nParticles)) ; ux=0.0_wp ; uy=0.0_wp ; uz=0.0_wp
        allocate(rv1(nParticles), rv2(nParticles), rv3(nParticles), rv4(nParticles))
        rv1=0.0_wp ; rv2=0.0_wp ; rv3=0.0_wp ; rv4=0.0_wp

        call init_random_seed()
    end if
    !----------------------------------------

    ! open the files
    call openFiles

end subroutine iniSimulation
!
!
!
!
subroutine openFiles

    use parameters_mod

    if (append) then
        open(23, file='out_md/out_largest_acceleration.dat', position='append', status='unknown')
        open(21, file='out_md/out_shortest_distance.dat', position='append', status='unknown')
        open(13, file='out_md/out_conservation.dat', position='append', status='unknown')
        open(15, file='out_md/out_movie.xyz', position='append', status='unknown')
        open(17, file='out_md/out_equil_movie.xyz', position='append', status='unknown')
        open(19, file='out_md/out_MD_phaseSpace.dat', access='stream', status='unknown')
    else
        open(23, file='out_md/out_largest_acceleration.dat', status='replace')
        open(21, file='out_md/out_shortest_distance.dat', status='replace')
        open(13, file='out_md/out_conservation.dat', status='replace')
        open(15, file='out_md/out_movie.xyz', status='replace')
        open(17, file='out_md/out_equil_movie.xyz', status='replace')
        open(19, file='out_md/out_MD_phaseSpace.dat', access='stream', status='replace')
    end if

end subroutine openFiles
!
!
!
!
subroutine closeFiles

    close(13)
    close(15)
    close(17)
    close(19)
    close(21)
    close(23)

end subroutine closeFiles
!
!
!
!
subroutine cleanup
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none

    ! deallocate arrays
    deallocate(headLCL)
    deallocate(listLCL)
    deallocate(sp)
    deallocate(mass)
    deallocate(charge)
    deallocate(rx)
    deallocate(ry)
    deallocate(rz)
    deallocate(vx)
    deallocate(vy)
    deallocate(vz)
    deallocate(ax)
    deallocate(ay)
    deallocate(az)
    deallocate(force)
    deallocate(forceFromSpecies)
    deallocate(f_priv)
    deallocate(forceFSp_priv)
    deallocate(listMTS)
    deallocate(isMTS)
    deallocate(forceMTS_internal)
    deallocate(forceMTS_external)
    deallocate(guiltyParticles)
    deallocate(guiltyParticles2)
    deallocate(guiltyCells)
    deallocate(a_tmp)
    if (allocated(friction)) deallocate(friction)
    if (allocated(avgKE)) deallocate(avgKE)
    if (allocated(ux)) deallocate(ux)
    if (allocated(uy)) deallocate(uy)
    if (allocated(uz)) deallocate(uz)
    if (allocated(rv1)) deallocate(rv1)
    if (allocated(rv2)) deallocate(rv2)
    if (allocated(rv3)) deallocate(rv3)
    if (allocated(rv4)) deallocate(rv4)

    ! close files
    call closeFiles

    ! turn off initialized flag
    initialized = .false.

end subroutine cleanup
!
!
!
!
subroutine movie
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    
    integer(i4b):: i
    integer(i4b):: fid
    if (thermostatOn) then
        fid = 17
    else
        fid = 15
    end if

    write(fid,*) nParticles
    write(fid,*)
    do i = 1, nParticles
        if(sp(i) == 1) then ;     write(fid,"(a1,1x,6(e13.6,1x))") 'A', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        elseif(sp(i) == 2) then ; write(fid,"(a1,1x,6(e13.6,1x))") 'B', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        elseif(sp(i) == 3) then ; write(fid,"(a1,1x,6(e13.6,1x))") 'C', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        elseif(sp(i) == 4) then ; write(fid,"(a1,1x,6(e13.6,1x))") 'D', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        elseif(sp(i) == 5) then ; write(fid,"(a1,1x,6(e13.6,1x))") 'E', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        elseif(sp(i) == 6) then ; write(fid,"(a1,1x,6(e13.6,1x))") 'F', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        elseif(sp(i) == 7) then ; write(fid,"(a1,1x,6(e13.6,1x))") 'G', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        elseif(sp(i) == 8) then ; write(fid,"(a1,1x,6(e13.6,1x))") 'H', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        elseif(sp(i) == 9) then ; write(fid,"(a1,1x,6(e13.6,1x))") 'I', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        else                    ; write(fid,"(a1,1x,6(e13.6,1x))") 'J', rx(i), ry(i), rz(i), vx(i), vy(i), vz(i)
        end if
    end do
    
end subroutine movie
!
!
!
!
subroutine conservation
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    
    kineticE = 0.5_wp*sum(mass*(vx**2 +vy**2 +vz**2))
    totalE = kineticE +potentialE

    write(13, '(i8,1x, 3(e13.6,1x))') iTime, totalE, kineticE, potentialE
    
end subroutine conservation
!
!
!
!
subroutine rescaleVel
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none
    
    real(wp):: scale, kinE
    
    kinE = 0.5_wp*sum(mass*(vx**2 +vy**2 +vz**2))
    scale = sqrt(nParticles*1.5_wp*kBolt*100000.0_wp/kinE)
    vx = scale*vx
    vy = scale*vy
    vz = scale*vz

end subroutine rescaleVel
!
!
!
!
subroutine outputForTaus
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none

    integer(i4b):: i, j

    do i = 1, nParticles
        write(100) vx(i)
        write(100) vy(i)
        write(100) vz(i)
        do j = 1, nSpecies
            write(100) forceFromSpecies(1,j,i)
            write(100) forceFromSpecies(2,j,i)
            write(100) forceFromSpecies(3,j,i)
        end do
    end do

end subroutine outputForTaus
!
!
!
!
subroutine writePhaseSpace
    use type_mod ; use parameters_mod ; use particles_mod ; implicit none

    integer(i4b):: i
    do i = 1, nParticles
        write(19) rx(i)
        write(19) ry(i)
        write(19) rz(i)
        write(19) vx(i)
        write(19) vy(i)
        write(19) vz(i)
    end do
    
end subroutine writePhaseSpace
!
!
!
!
subroutine init_random_seed()

      integer :: i, n, clock
      integer, dimension(:), allocatable :: seed

      call random_seed(size = n)
      allocate(seed(n))

      call system_clock(count=clock)

      seed = clock + 37 * (/ (i - 1, i = 1, n) /)
      call random_seed(put = seed)

      deallocate(seed)

end
