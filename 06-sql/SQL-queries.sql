use wine_project;

######### Import data #########

# Used the Import Wizard to import the two tables

SELECT * FROM wine_data; 
SELECT * FROM grapes;

######### Wineries #########

# Obtain a list of the wineries

SELECT DISTINCT(winery) FROM wine_data;

######### Price #########

# Average price per type of wine

SELECT type_wine, ROUND(AVG(price), 1) AS avg_price
FROM wine_data
GROUP BY type_wine
ORDER BY avg_price DESC;

# Average price per region (Geographical Indication)

SELECT region_gi, ROUND(AVG(price), 1) AS avg_price
FROM wine_data
GROUP BY region_gi
ORDER BY avg_price DESC;

# Which 5 grapes varieties give the most expensive wines? 

SELECT G.grapes, ROUND(AVG(W.price), 1) AS avg_price
FROM grapes G
JOIN wine_data W	
USING (wine)
GROUP BY G.grapes
ORDER BY avg_price DESC
LIMIT 5;

# Regions where the price is below average

SELECT region, ROUND(AVG(price), 1) AS avg_price 
FROM wine_data
WHERE price < (SELECT AVG(price)
FROM wine_data)
GROUP BY region
ORDER BY avg_price ASC;

######### Type of agriculture #########

# Count the number of wines in each type of agriculture 

SELECT type_agriculture, COUNT(*) AS num_wines
FROM wine_data
GROUP BY type_agriculture
ORDER BY num_wines DESC;

######### Grapes #########

# Select only the wines made with Garnacha grapes

SELECT * 
FROM wine_data W
JOIN grapes G
USING (wine)
WHERE G.grapes = "Garnacha";

