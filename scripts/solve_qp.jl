# Copyright 2021 The FirstOrderLp Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is an interface to FirstOrderLp for solving a single QP or LP and writing
# the solution and solve statistics to a file. Run with --help for a pretty
# description of the arguments. See the comments on solve_instance_and_output
# for a description of the output formats.

import FirstOrderLp
include("input_output.jl")

function main()
  parsed_args = parse_command_line()
  parameters = process_args(parsed_args)

  solve_instance_and_output(
    parameters,
    parsed_args["output_dir"],
    parsed_args["instance_path"],
    parsed_args["redirect_stdio"],
    parsed_args["transform_bounds_into_linear_constraints"],
    parsed_args["fixed_format_input"],
  )
end

main()
