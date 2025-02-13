function main()
  x1 = [1, 1, 1]
  x2 = [2, 2, 2]
  x3 = [3, 3, 3]

  queue = Vector{Vector{Float64}}()

  push!(queue, x1)
  push!(queue, x2)

  # Printing the elements: 
  print("Queue 1: ")
  for element in queue
    print(element, " ")
  end
  print("\n")

  push!(queue, x3)
  popfirst!(queue)

  # Printing the elements: 
  print("Queue 2: ")
  for element in queue
    print(element, " ")
  end
  print("\n")    
  
  println(last(queue))
end

main()