package com.collabera.stock;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import com.collabera.stock.service.StockService;

@SpringBootApplication
public class StockProjectApplication {
	
	public static void main(String[] args) {
		SpringApplication.run(StockProjectApplication.class, args);
	}

}
