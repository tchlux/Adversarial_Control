! Waypoint distance fortran 

  ! ! Subroutine for projecting a point onto a line
  ! SUBROUTINE point_to_line(p1, p2, point, new_point)
  !   ! Subroutine for projecting a point onto a line
  !   REAL, DIMENSION(3), INTENT(IN) :: point, p1, p2
  !   REAL, DIMENSION(3), INTENT(OUT) :: new_point
  !   new_point = p2 - p1
  !   new_point = p1 + DOT_PRODUCT(point-p1, new_point) / &
  !        DOT_PRODUCT(new_point, new_point) * new_point
  ! END SUBROUTINE point_to_line




PROGRAM waypoint_distance
  IMPLICIT NONE
  CHARACTER(LEN=32) :: arg
  REAL :: min_dist_from_line
  REAL, DIMENSION(3,28) :: waypoints
  REAL, DIMENSION(3) :: point, projected_point
  REAL :: line_segment_length, dist_from_p1, dist_from_p2, dist_from_line
  INTEGER :: pt_index
  ! Check the command line arguments
  IF (IARGC() .EQ. 0) THEN
     PRINT *, ''
     PRINT*, 'Summary:'
     PRINT*, '  This is a single-execution fortran script designed to return the'
     PRINT*, 'distance of a given point to the nearest path-line. The path-line is'
     PRINT*, 'defined as the 3 dimensional line through space given by a predefined'
     PRINT*, 'list of waypoints.'
     PRINT*, ''
     PRINT*, 'Usage:'
     PRINT*, '  $ gfortran -o waypoint_distance waypoint_distance.f90'
     PRINT*, '  $ ./waypoint_distance <float> <float> <float>'
     PRINT*, ''
     PRINT*, 'Where the 3 floats are (x,y,z) coordinates respectively.'
     RETURN
  ELSE IF (IARGC() .NE. 3) THEN
     PRINT *, 'ERROR: Must pass in exactly 3 command line arguments.'
     RETURN
  ELSE
     ! Read in the triple that represents the point
     CALL GETARG(1, arg)
     READ(arg,*) point(1)
     CALL GETARG(2, arg)
     READ(arg,*) point(2)
     CALL GETARG(3, arg)
     READ(arg,*) point(3)
  END IF

  ! Initialize the waypoints
  waypoints = RESHAPE((/ &
       179.30, 2.90, 98.70, &
       172.30, 2.90, 117.70, &
       152.30, 2.90, 140.40, &
       132.80, 2.90, 150.90, &
       112.70, 2.90, 157.50, &
       92.00, 2.90, 160.00, &
       70.40, 2.90, 156.90, &
       45.30, 2.90, 151.80, &
       15.90, 2.90, 139.10, &
       -10.00, 2.90, 126.60, &
       -43.50, 2.90, 105.50, &
       -78.30, 2.90, 77.50, &
       -123.40, 2.90, 33.40, &
       -158.30, 2.90, -18.70, &
       -175.50, 2.90, -66.50, &
       -169.60, 2.90, -115.10, &
       -146.00, 2.90, -141.30, &
       -107.80, 2.90, -154.70, &
       -39.80, 2.90, -158.70, &
       47.20, 2.90, -148.10, &
       107.80, 2.90, -133.50, &
       129.00, 2.90, -111.70, &
       120.00, 2.90, -72.70, &
       84.20, 2.90, -20.70, &
       77.20, 2.90, 3.90, &
       90.80, 2.90, 22.80, &
       156.80, 2.90, 60.30, &
       176.10, 2.90, 79.60 &
       /) , SHAPE(waypoints))

  ! Initialize minimum distance
  min_dist_from_line = HUGE(min_dist_from_line)

  ! Find the line segment that is closest to this point
  DO pt_index = 1, SIZE(waypoints,2)-1
     ! Projected point
     CALL point_to_line( waypoints(:,pt_index), waypoints(:,pt_index+1), &
          point, projected_point )
     ! Line segment length
     line_segment_length = &
          SQRT(SUM((waypoints(:,pt_index) - waypoints(:,pt_index+1))**2))
     ! Distance from p1
     dist_from_p1 = SQRT(SUM((projected_point - waypoints(:,pt_index))**2))
     ! Distance from p2
     dist_from_p2 = SQRT(SUM((projected_point - waypoints(:,pt_index+1))**2))
     ! Distance from line
     dist_from_line = SQRT(SUM((projected_point - point)**2))
     ! Check condition
     IF (MIN(dist_from_p1, dist_from_p2) .LE. line_segment_length) THEN
        IF (dist_from_line .LE. min_dist_from_line) &
             min_dist_from_line = dist_from_line
     END IF
  END DO

  PRINT *, min_dist_from_line

CONTAINS

  ! Subroutine for projecting a point onto a line
  SUBROUTINE point_to_line(p1, p2, point, new_point)
    ! Subroutine for projecting a point onto a line
    REAL, DIMENSION(3), INTENT(IN) :: point, p1, p2
    REAL, DIMENSION(3), INTENT(OUT) :: new_point
    new_point = p2 - p1
    new_point = p1 + DOT_PRODUCT(point-p1, new_point) / &
         DOT_PRODUCT(new_point, new_point) * new_point
  END SUBROUTINE point_to_line

END PROGRAM waypoint_distance

! gfortran -o waypoint_distance waypoint_distance.f90 && ./waypoint_distance 1 2 3
