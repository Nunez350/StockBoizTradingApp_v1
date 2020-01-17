package com.collabera.stock.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.collabera.stock.model.Stock;

public interface StocksRepository extends JpaRepository<Stock, Integer>{

}
