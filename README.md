# SIREKAP Investigation Tools

This repository contains a set of tools written in python to investigate the SIREKAP data through their public APIs.
A full bulk download of data and images is possible using the tools, but will use a lot of bandwidth and storage (~2TB).

Initially I was uncomfortable with releasing the tools as it increases the risk of API abuse as it lowers barrier to entry.
However, I believe that the benefits of releasing the tools outweigh the risks. That being said, I urge users using
the data retrieval tool to consider the following:
- Don't set concurrent requests too high.
- Use of the tool should be restricted to outside Indonesia's waking hours.
- A full bulk download should be limited to once a day, the tool records the GET path needed to refresh each polling station.

Misuse of the tool could lead to the API being taken down, which would be a loss to the public.

## Data Retrieval Tool Output Format
The data retrieval tool outputs the numeric data into a JSON file as a JSON object/dictionary, in the following format:
```json
{
	...,
	"PATH_OF_STATION": {
		"data_url": "URL_TO_DATA",
		...rest following as per API response data for that station
	},
	...
}
```

The image retrieval function outputs the images of the second page of the C1 form into the chosen directory.
The images are named as their ID in the API response.