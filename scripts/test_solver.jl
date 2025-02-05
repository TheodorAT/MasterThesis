import FirstOrderLp
include("input_output.jl")

function main()
  # TODO: Add possibility to run on multiple instances from command line. 
  parsed_args = parse_command_line()
  parameters = process_args(parsed_args)
  
  max_memory_list = parsed_args["max_memory"]
  for max_memory in parsed_args["max_memory"]  
    dwifob_parameters = FirstOrderLp.DwifobParameters(
      max_memory,
    )
    new_output_dir= parsed_args["output_dir"] * "_m=$(max_memory)"  
    println("new_output_dir: ", new_output_dir)

    solve_instance_and_output(
      parameters,
      new_output_dir,
      parsed_args["instance_path"],
      parsed_args["redirect_stdio"],
      parsed_args["transform_bounds_into_linear_constraints"],
      parsed_args["fixed_format_input"],
      dwifob_parameters,
    ) 
  end
end

main()