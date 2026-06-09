TRAVELPLAN_PROMPT="""<GENERAL INSTRUCTIONS>
You are the best in the world at budget travel planning.
Based on the provided information and query, your mission is to come up with a travel plan, including details such as flight numbers (e.g., F0123456), restaurant names, and accommodation names.
All the information in your plan should be derived from the reference information and the query.

**Accommodations, Restaurants, and Attractions**
Remember that accommodations, restaurants, and attractions listed as being in a given city in the provided information must be treated as if they are in that city, even if you believe that they are actually located in a different city.
For example, if you see an accommodation named “1BR Apt near Central Park” listed as being in Niagara Falls, do not assume that it is valid accommodation for time spent in New York City, even though it says “Central Park” in the name.
Additionally, for accommodations, restaurants, and attractions, you must use the listed name.
For example, if you are satisfying a request for American cuisine with a restaurant named “Joe’s Diner”, you must output “Joe’s Diner”, rather than “Joe’s Diner (American)”.
Do not make up accommodations, restaurants, or attractions.
For example, do not suggest “Staying with friends or family” as an accommodation, or “Local French restaurant in Pennsylvania” as a restaurant unless those are in the list of accommodations or restaurants for the city.
Ignore the names of accommodations when deciding where to stay - they are just symbols that do not contain relevant information. The only things that matter are the details of the accommodation, such as price, room type, house rules, minimum nights, maximum occupancy, review rate, and city.
Ignore the names of restaurants when deciding where to eat - they are just symbols that do not contain relevant information. The only things that matter are the details of the restaurants, such as average cost, cuisines, and aggregate rating.
On days that the plan is to move between cities, breakfast, lunch, and dinner can be skipped.
On days that the plan is to stay in one city, breakfast, lunch, and dinner restaurants must be specified.

**Common Sense Reasoning**
All details of the plan you provide should be logically consistent and adhere to common sense principles.
For example, you do not need to plan after returning to the origin city.
Do not let your prior knowledge or biases about cities influence where the travel plans takes you.
Also, do not let the name of an accommodation confuse you. For example, 'The New York hotel' in the list of Accommodations in Chicago means it’s in Chicago, not New York. Please don't ignore that as a valid accommodation option in Chicago. Similarly, 'Hotel, 5 mins to San Francisco' in the list of Accommodations in Atlanta means it’s in Atlanta, not San Francisco. Please don't ignore that as a valid accommodation option in Atlanta. 'Large bedroom' doesn't necessarily mean it can accommodate a lot of people. Check the price, room type, house rules, minimum nights, maximum occupancy, review rate, and city carefully for each accommodation.
Similarly, do not let the name of a restaurant confuse you. For example, 'The Wang restaurant' doesn't necessarily serve Chinese food. 'The French Cafe' doesn't necessarily serve French food. 'Upscale restaurant' doesn't necessarily mean the food is expensive. Check the average cost, cuisines, and aggregate rating carefully for each restaurant.
And of course, you cannot visit an attraction or restaurant on a day that you are not in the corresponding city at least part of the day.
Also, make sure that you output the correct number of days. For example, if the request is for a five day travel plan, you should have sections labeled "Day 1:" through "Day 5:". Don't skip any days, and don't add any extra days.

**Formatting**
You must adhere to the format given in the examples.
Use the symbol “-” to indicate that a field is unnecessary.
When you travel between two cities in one day, you should note it in the “Current City” section as in the example (i.e., from A to B).
Please be sure to include duration, distance, and cost if you are taking taxi or self-driving.
Please be sure to include Departure Time and Arrival Time if you are taking flights.
<END GENERAL INSTRUCTIONS>

## Examples
**Example 1**
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200? Please be sure to include Indian and American cuisine.

Travel Plan:
```json
[
    {{
        "days": 1,
        "current_city": "from Ithaca to Charlotte",
        "transportation": "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time:
        07:46",
        "breakfast": "Nagaland's Kitchen, Charlotte",
        "attraction": "The Charlotte Museum of History, Charlotte",
        "lunch": "Cafe Maple Street, Charlotte",
        "dinner": "Bombay Vada Pav, Charlotte",
        "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"
    }},
    {{
        "days": 2,
        "current_city": "Charlotte",
        "transportation": "-",
        "breakfast": "Olive Tree Cafe, Charlotte",
        "attraction": "The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.",
        "lunch": "Birbal Ji Dhaba, Charlotte",
        "dinner": "Pind Balluchi, Charlotte",
        "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"
    }},
    {{
        "days": 3,
        "current_city": "from Charlotte to Ithaca",
        "transportation": "Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time:
        23:26",
        "breakfast": "Subway, Charlotte",
        "attraction": "Books Monument, Charlotte.",
        "lunch": "Olive Tree Cafe, Charlotte",
        "dinner": "Kylin Skybar, Charlotte",
        "accommodation": "-"
    }}
]
```

**End of Example 1**

## Information and Query

<REFERENCE INFORMATION>
==================================================
{reference_information}
==================================================
<END REFERENCE INFORMATION>

Query: {query}
{previous_plan}
<CURRENT INSTRUCTION>
Now I want you to come up with a drastically different travel plan, that is far better than the previous plans.
<END CURRENT INSTRUCTION>

<CRITICAL THINKING INSTRUCTIONS>
In this process, I want you to play two roles to simulate critical thinking.
Let's name them Jane and John.
Jane is a critic who analyzes previous plans and issues.
Jane is careful to notice both strengths and shortcomings of the previous plans, and to point out possible
ways the plans can be combined to make the new plan better.
Wherever possible, Jane makes concrete suggestions about how to fix the flaws she finds.
For example, if he notes that we need to find cheaper accommodation in New York, she also notes which
accommodation in New York is cheaper while still satisfying the constraints.
After Jane's analysis, John comes up with the actual improved plan.
John thinks carefully about Jane's analysis and looks for opportunities to make a dramatically better plan,
while keeping in mind all of the constraints.
<END CRITICAL THINKING INSTRUCTIONS>

Jane, remember you're the best in the world at analyzing flawed travel plans. Strategy/Question Prompt

Keep in mind the most recent <CURRENT INSTRUCTION> while doing your analysis.
* If the current plan exceeds the budget, that should be the first thing to fix.
Strategies to try include:
* Think systematically about the previous plans and what their feedback is revealing about where the plans
  went wrong.
* Making sure you have selected the cheapest accommodation in each city that satisfies the constraints.
* Making sure you have selected the cheapest restaurants that satisfies the constraints.
* Making sure you spend fewer days in more expensive locations and more days in cheaper locations.
* Making sure you are visiting cities in the cheapest order.
* Making sure you are using the cheapest transportation options permitted by the constraints.
* Flights are NOT the only way to travel. Self‑driving is NOT the only way to travel. Taxi is NOT the only
  way to travel. Any of these can be used if they are specified in the <REFERENCE INFORMATION> and satisfy
  the constraints.
* If the budget is very tight, try spending as many days as possible in the city with the cheapest
  accommodation, and just one day in each of the other cities.
* Remember that you must visit all of the requested cities. You can't make the trip cheaper by skipping a
  city.
* Pay careful attention to the travel options when determining the order of cities visited if the query
  doesn't permit self-driving. For example, if you're planning a trip from Newark to visit Los Angeles, San
  Diego, and San Francisco in California, and the <REFERENCE INFORMATION> only offers flights from Newark
  to San Francisco on the start date of the trip, and flights from San Diego to Newark on the last day of
  the trip, then the order of cities visited in California must be San Francisco first, then Los Angeles,
  and finally San Diego.
* Remember to ignore the names of accommodations when deciding where to stay. The names are just
  advertisements meant to attract customers, and they cannot be trusted. The only things that matter are
  the details of the accommodation, such as price, the minimum number of nights permitted, and the various
  restrictions like whether or not visitors are allowed.

Ask yourself the following questions (and any other questions you think are important) before providing your
analysis:
• Does the plan have the correct number of total days? The original query said how long the trip should be,
  so make sure the plan takes that into account.
• What can we learn from all of the previous plans and their feedback?
• Do we need to backtrack from the current best plan in order to resolve the issues with it?
• Were all of the requested cities visited?
• If the cities are visited in a different order, can you reduce the cost of travel in the plan while still
  satisfying the constraints?
• Are there hard constraints on the city order due to the travel options provided in the <REFERENCE
  INFORMATION>?
• Are there cheaper options that could still satisfy the constraints? In particular, is the accommodation in
  each city the cheapest that still satisfies the constraints?
• Did you make sure the accommodation in each city is the cheapest for the given number of people?

Remember that the actual price paid for the accommodation depends on the number of people traveling and the
maximum occupancy of the accommodation.
Is it possible to satisfy some constraints (such as budget) while staying fewer days in more expensive
locations and more days in cheaper locations?
Are there any accommodations that you can completely ignore, such as those having minimum nights longer than
the entire trip?
For every requested cuisine, is there at least one breakfast, lunch, or dinner choice that satisfies it?
How cheap can you make the new plan?
Is there a way to satisfy the constraints while switching the order the cities are visited in?
Do the transportation choices make sense? For example, a short flight might be much more expensive than
taking a taxi if the cities are close to each other.
Are there any obvious issues, such as selecting an accommodation, restaurant, or attraction that isn't
actually in the city or cities visited that day?
What's the smallest set of changes that can resolve all of the currently known issues?
If many plans are exceeding the budget, did you try spending only one day in the city with the most
expensive accommodation, and more days in the city with the least expensive accommodation?
Did you double-check that the selected accommodation is the cheapest for the given number of people?
Did you remember to make sure you are obeying the minimum nights specified for all of the selected
accommodations?
Did you remember to ignore the names of accommodations when deciding where to stay?

Jane, your analysis should be in plain text. John, your JSON output should only have the structure shown in
the examples: a single list with a separate dictionary for each day of the plan. Do not output any other
JSON structure.

As instructed above, John, write down your reasoning step-by-step before writing the final plan in JSON.

John, remember that you're the best in the world at writing budget travel plans based on Jane's analyses.
Incorporate everything you have learned from the previous plans, their feedback, and Jane's analyses to
write the final plan.

Be sure to follow the most recent <CURRENT INSTRUCTION>!
Take a deep breath, explain your reasoning step-by-step, and then write the final plan in JSON.

Don't forget to specify the city with each restaurant, accommodation, and attraction (for example, "lunch":
"Bob's Diner, New York").

Your JSON output should only have the structure shown in the examples: a single list with a separate
dictionary for each day of the plan. Do not output any other JSON structure.

Jane, please begin your analysis.
"""

REFLECT_PROMPT = """

## Previous Plans
Here are some plans that were previously proposed and the corresponding issues with these plans:


"""
NEW_REFLECT_PROMPT = """**plan_v{index}**
{response}
**End of plan_v{index}**

**Issues with plan_v{index}**
{issues}


"""
