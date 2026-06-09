NATURALPLAN_MEETING_PROMPT="""<GENERAL INSTRUCTIONS> You are the best in the world at meeting planning. You will be provided with the initial location and time, your friends' schedules, how long you would like to meet with each one of them, and the travel distances between different locations. Your mission is to come up with a plan that allows you to meet as many friends as possible by considering these constraints.
The final plan format must strictly adhere to what the examples below show, beginning from "Meeting Plan:" and ending with "```".

DO NOT add any comments to the output JSON.
<END GENERAL INSTRUCTIONS>

Examples
Here are a few example tasks and solutions:
Example 1
<EXAMPLE REQUIREMENTS>
TASK: You are visiting San Francisco for the day and want to meet as many friends as possible. Solve the problem by considering various different schedules and picking the best one to optimize your goals.

Travel distances (in minutes):
Marina District to Alamo Square: 15.
Marina District to Fisherman's Wharf: 10.
Marina District to Union Square: 16.
[...]

You arrive at Marina District at 9:00AM. Anthony will be at Alamo Square from 11:00AM to 12:45PM. You'd like to meet Anthony for a minimum of 45 minutes. Daniel will be at Fisherman's Wharf from 2:00PM to 6:15PM. You'd like to meet Daniel for a minimum of 90 minutes. Jessica will be at Union Square from 12:15PM to 9:30PM. You'd like to meet Jessica for a minimum of 105 minutes. Barbara will be at Embarcadero from 7:00AM to 10:15AM. You'd like to meet Barbara for a minimum of 120 minutes. Thomas will be at Financial District from 6:00PM to 10:00PM. You'd like to meet Thomas for a minimum of 105 minutes. Mark will be at Nob Hill from 10:30AM to 1:30PM. You'd like to meet Mark for a minimum of 120 minutes. Timothy will be at Haight-Ashbury from 9:15PM to 10:15PM. You'd like to meet Timothy for a minimum of 60 minutes. Stephanie will be at Richmond District from 7:45PM to 8:30PM. You'd like to meet Stephanie for a minimum of 45 minutes. Joseph will be at The Castro from 12:45PM to 2:45PM. You'd like to meet Joseph for a minimum of 30 minutes.
<END EXAMPLE REQUIREMENTS>

<EXAMPLE SOLUTION> Meeting Plan: ```json [ { "plan": [ "You start at Marina District at 9:00AM", "You travel to Alamo Square in 15 minutes and arrive at 9:15AM", "You wait until 11:00AM", "You meet Anthony for 45 minutes from 11:00AM to 11:45AM", "You travel to The Castro in 8 minutes and arrive at 11:53AM", "You wait until 12:45PM", "You meet Joseph for 30 minutes from 12:45PM to 1:15PM", "You travel to Fisherman's Wharf in 24 minutes and arrive at 1:39PM", "You wait until 2:00PM", "You meet Daniel for 90 minutes from 2:00PM to 3:30PM", "You travel to Union Square in 13 minutes and arrive at 3:43PM", "You meet Jessica for 105 minutes from 3:43PM to 5:28PM", "You travel to Financial District in 9 minutes and arrive at 5:37PM",
    "You wait until 6:00PM",
    "You meet Thomas for 105 minutes from 6:00PM to 7:45PM",
    "You travel to Haight-Ashbury in 19 minutes and arrive at 8:04PM",
    "You wait until 9:15PM",
    "You meet Timothy for 60 minutes from 9:15PM to 10:15PM"
]
}
]
…

<END EXAMPLE SOLUTION> **End of Example 1**
Example 2
[...]
End of Example 5

Task and Information
Now I would like you to solve the following meeting planning task similar to the examples above.

<REQUIREMENTS> TASK: You are visiting San Francisco for the day and want to meet as many friends as possible. Solve the problem by considering various different schedules and picking the best one to optimize your goals.
Travel distances (in minutes):
Union Square to The Castro: 17.
Union Square to North Beach: 10.
Union Square to Embarcadero: 11.
[...]

You arrive at Union Square at 9:00AM. Melissa will be at The Castro from 8:15PM to 9:15PM. You'd like to meet Melissa for a minimum of 30 minutes. Kimberly will be at North Beach from 7:00AM to 10:30AM. You'd like to meet Kimberly for a minimum of 15 minutes. Joseph will be at Embarcadero from 3:30PM to 7:30PM. You'd like to meet Joseph for a minimum of 75 minutes. Barbara will be at Alamo Square from 8:45PM to 9:45PM. You'd like to meet Barbara for a minimum of 15 minutes. Kenneth will be at Nob Hill from 12:15PM to 5:15PM. You'd like to meet Kenneth for a minimum of 105 minutes. Joshua will be at Presidio from 4:30PM to 6:15PM. You'd like to meet Joshua for a minimum of 105 minutes. Brian will be at Fisherman's Wharf from 9:30AM to 3:30PM. You'd like to meet Brian for a minimum of 45 minutes. Steven will be at Mission District from 7:30PM to 9:00PM. You'd like to meet Steven for a minimum of 90 minutes. Betty will be at Haight-Ashbury from 7:00PM to 8:30PM. You'd like to meet Betty for a minimum of 90 minutes.
<END REQUIREMENTS>

"""