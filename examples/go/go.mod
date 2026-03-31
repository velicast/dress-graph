module dress-graph-examples

go 1.21

require (
	github.com/velicast/dress-graph/go v0.0.0
	github.com/velicast/dress-graph/go/cuda v0.0.0
	github.com/velicast/dress-graph/go/mpi v0.0.0
	github.com/velicast/dress-graph/go/mpi/cuda v0.0.0
	github.com/velicast/dress-graph/go/mpi/omp v0.0.0
	github.com/velicast/dress-graph/go/omp v0.0.0
)

replace (
	github.com/velicast/dress-graph/go => ../../go
	github.com/velicast/dress-graph/go/cuda => ../../go/cuda
	github.com/velicast/dress-graph/go/mpi => ../../go/mpi
	github.com/velicast/dress-graph/go/mpi/cuda => ../../go/mpi/cuda
	github.com/velicast/dress-graph/go/mpi/omp => ../../go/mpi/omp
	github.com/velicast/dress-graph/go/omp => ../../go/omp
)
